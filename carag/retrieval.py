"""Hybrid, intent-aware, adaptive evidence retrieval.

Scoring (per evidence sentence):

    score = w_sem * max(cosine, 0) + w_lex * BM25_normalized + bonus

``bonus`` rewards evidence types the question asks for (causal, comparison,
numeric, limitation) *only in proportion to how many query key terms the
sentence contains*. A sentence containing "because" but none of the question's
terms therefore receives no bonus.

Adaptive selection:
  1. Rank all sentences; keep the top ``initial_k`` that score at least
     ``relative_cutoff`` x best score (refinement: drops weak tail evidence).
  2. Check sufficiency: best score, key-term coverage of the selected set,
     and whether an evidence type required by the question is present.
  3. If insufficient, expand: (a) search for missing key terms,
     (b) add neighbouring sentences of the best hits (to recover
     explanations split across sentences), (c) widen k and relax the cutoff.
  4. Near-duplicates are suppressed throughout.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import RetrievalConfig
from .index import EvidenceIndex
from .query import QueryAnalysis, analyze_question
from .schema import ScoredEvidence
from .text_utils import content_terms

_INTENT_FEATURES = ("causal", "comparison", "numeric", "limitation")


@dataclass
class RetrievalResult:
    question: str
    analysis: QueryAnalysis
    evidence: list[ScoredEvidence]
    sufficient: bool
    reasons: list[str] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    expanded: bool = False
    refined: bool = False

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "sufficient": self.sufficient,
            "reasons": self.reasons,
            "trace": self.trace,
            "expanded": self.expanded,
            "refined": self.refined,
            "intents": sorted(self.analysis.intents),
            "key_terms": self.analysis.key_terms,
            "evidence": [e.to_dict() for e in self.evidence],
        }


class HybridScorer:
    """Computes semantic, lexical and intent components for every indexed sentence."""

    def __init__(self, index: EvidenceIndex, config: RetrievalConfig):
        self.index = index
        self.config = config

    def score(self, text: str, key_terms: list[str], intents: set[str] | None = None,
              query_vector: np.ndarray | None = None) -> dict[str, np.ndarray]:
        if query_vector is None:
            query_vector = self.index.embedder.encode([text])[0]
        semantic = np.clip(self.index.semantic_scores(query_vector), 0.0, 1.0)
        lexical = self.index.lexical_scores(key_terms)
        n = len(self.index)
        coverage = np.array([self.index.term_coverage(i, key_terms) for i in range(n)], dtype=np.float32)
        bonus = np.zeros(n, dtype=np.float32)
        wanted = [f for f in _INTENT_FEATURES if intents and f in intents]
        if wanted:
            has_feature = np.array(
                [any(self.index.unit_features[i][f] for f in wanted) for i in range(n)], dtype=np.float32)
            bonus = self.config.intent_bonus * has_feature * coverage
        total = self.config.semantic_weight * semantic + self.config.lexical_weight * lexical + bonus
        return {"total": total, "semantic": semantic, "lexical": lexical,
                "coverage": coverage, "bonus": bonus}

    def make_evidence(self, i: int, scores: dict[str, np.ndarray], found_by: str) -> ScoredEvidence:
        return ScoredEvidence(
            unit=self.index.units[i],
            score=round(float(scores["total"][i]), 4),
            semantic=round(float(scores["semantic"][i]), 4),
            lexical=round(float(scores["lexical"][i]), 4),
            coverage=round(float(scores["coverage"][i]), 4),
            bonus=round(float(scores["bonus"][i]), 4),
            found_by=found_by,
        )


class AdaptiveRetriever:
    def __init__(self, index: EvidenceIndex, config: RetrievalConfig | None = None):
        self.index = index
        self.config = config or RetrievalConfig()
        self.scorer = HybridScorer(index, self.config)

    # -- simple rankers (also used as baselines) ---------------------------------
    def rank(self, text: str, top_n: int, mode: str = "hybrid",
             intents: set[str] | None = None) -> list[ScoredEvidence]:
        """Plain top-n ranking. mode: 'hybrid' | 'semantic' | 'lexical'."""
        if not len(self.index):
            return []
        scores = self.scorer.score(text, content_terms(text), intents)
        key = {"hybrid": "total", "semantic": "semantic", "lexical": "lexical"}[mode]
        order = np.argsort(-scores[key], kind="stable")[:top_n]
        return [self.scorer.make_evidence(int(i), scores, f"top-{top_n} {mode}") for i in order]

    # -- adaptive retrieval -------------------------------------------------------
    def _is_redundant(self, i: int, chosen: list[int]) -> bool:
        if not chosen:
            return False
        sims = self.index.embeddings[chosen] @ self.index.embeddings[i]
        return bool(sims.max() >= self.config.redundancy_threshold)

    def _sufficiency(self, chosen: list[int], scores, analysis: QueryAnalysis) -> tuple[bool, list[str]]:
        cfg = self.config
        reasons = []
        if not chosen:
            return False, ["No evidence passed the minimum relevance score."]
        best = float(max(scores["total"][i] for i in chosen))
        if best < cfg.sufficiency_score:
            reasons.append(f"Best evidence score {best:.2f} is below {cfg.sufficiency_score:.2f}.")
        terms = analysis.key_terms
        if terms:
            covered = {t for t in terms for i in chosen if t in self.index.unit_terms[i]}
            coverage = len(covered) / len(terms)
            if coverage < cfg.sufficiency_coverage:
                missing = [t for t in terms if t not in covered]
                reasons.append(f"Only {coverage:.0%} of key terms found (missing: {', '.join(missing)}).")
        for feature in ("causal", "numeric"):
            if feature in analysis.intents and not any(
                    self.index.unit_features[i][feature] and scores["coverage"][i] > 0 for i in chosen):
                reasons.append(f"The question asks for {feature} evidence, but none of the selected evidence "
                               f"is {feature} evidence about the question's subject.")
        return not reasons, reasons

    def _select(self, order, scores, chosen: list[int], limit: int, relative: float, found_by: str,
                sources: dict[int, str]) -> tuple[list[int], int]:
        cfg = self.config
        best = float(scores["total"][order[0]]) if len(order) else 0.0
        cutoff = max(cfg.absolute_min_score, relative * best)
        dropped = 0
        for i in order:
            i = int(i)
            if len(chosen) >= limit:
                break
            if i in chosen:
                continue
            if scores["total"][i] < cutoff and len(chosen) >= cfg.min_k:
                dropped += 1
                continue
            if scores["total"][i] < cfg.absolute_min_score:
                continue
            if self._is_redundant(i, chosen):
                continue
            chosen.append(i)
            sources[i] = found_by
        return chosen, dropped

    def retrieve(self, question: str, analysis: QueryAnalysis | None = None) -> RetrievalResult:
        cfg = self.config
        analysis = analysis or analyze_question(question)
        if not len(self.index):
            return RetrievalResult(question, analysis, [], False, ["No sources have been ingested."])

        query_vector = self.index.embedder.encode([question])[0]
        scores = self.scorer.score(question, analysis.key_terms, analysis.intents, query_vector)
        order = np.argsort(-scores["total"], kind="stable")[: cfg.max_k * 4]
        found: dict[int, str] = {}
        trace: list[dict] = []

        chosen, dropped = self._select(order, scores, [], cfg.initial_k, cfg.relative_cutoff, "initial", found)
        trace.append({"round": 0, "action": "initial hybrid retrieval",
                      "kept": [self.index.units[i].evidence_id for i in chosen],
                      "cutoff_dropped": dropped})
        refined = dropped > 0
        sufficient, reasons = self._sufficiency(chosen, scores, analysis)
        expanded = False

        for round_number in range(1, cfg.max_expansion_rounds + 1):
            if sufficient:
                break
            expanded = True
            before = set(chosen)
            actions = []

            # (a) Target key terms that are still missing from the evidence set.
            missing = [t for t in analysis.key_terms
                       if not any(t in self.index.unit_terms[i] for i in chosen)]
            if missing:
                lexical = self.index.lexical_scores(missing)
                boosted = dict(scores)
                boosted["total"] = scores["total"] + cfg.lexical_weight * lexical
                sub_order = np.argsort(-boosted["total"], kind="stable")[: cfg.max_k * 2]
                chosen, _ = self._select(sub_order, scores, chosen, cfg.max_k, cfg.relative_cutoff - 0.1,
                                         f"expansion {round_number}: missing terms", found)
                actions.append(f"searched for missing terms: {', '.join(missing)}")

            # (b) Neighbouring sentences of the strongest evidence.
            for i in list(chosen[:2]) if cfg.expand_neighbors else []:
                for j in self.index.neighbors(i):
                    if j in chosen or len(chosen) >= cfg.max_k:
                        continue
                    wanted = [f for f in ("causal", "numeric", "comparison") if f in analysis.intents]
                    has_type = any(self.index.unit_features[j][f] for f in wanted)
                    if scores["total"][j] >= cfg.absolute_min_score or has_type:
                        if not self._is_redundant(j, chosen):
                            chosen.append(j)
                            found[j] = f"expansion {round_number}: neighbour of {self.index.units[i].evidence_id}"
            if cfg.expand_neighbors:
                actions.append("added neighbouring sentences of top evidence")

            # (c) Widen the candidate pool with a relaxed cutoff.
            chosen, _ = self._select(order, scores, chosen, cfg.max_k, cfg.relative_cutoff - 0.15 * round_number,
                                     f"expansion {round_number}: widened", found)
            actions.append("widened k and relaxed the relative cutoff")

            added = [self.index.units[i].evidence_id for i in chosen if i not in before]
            trace.append({"round": round_number, "action": "; ".join(actions), "added": added,
                          "reasons": reasons})
            sufficient, reasons = self._sufficiency(chosen, scores, analysis)
            if not added:
                break

        chosen = sorted(chosen, key=lambda i: -scores["total"][i])[: cfg.max_k]
        evidence = [self.scorer.make_evidence(i, scores, found.get(i, "initial")) for i in chosen]
        return RetrievalResult(question, analysis, evidence, sufficient, reasons, trace,
                               expanded=expanded, refined=refined)
