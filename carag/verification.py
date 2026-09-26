"""Claim-level verification against ingested evidence using an NLI cross-encoder.

Labels (applied in this order):

* SUPPORTED - a topically relevant evidence passage entails the full claim
  (P(entailment) >= support_threshold) and every number in the claim appears in
  that passage, and no relevant passage contradicts it.
* CONTRADICTED - a single, closely relevant sentence contradicts the claim
  (P(contradiction) >= contradiction_threshold, similarity >=
  contradiction_relevance_threshold) and none supports it, or a
  component of a causal claim is contradicted.
* UNCERTAIN - passages of comparable strength (relevance x NLI probability)
  both support and contradict the claim (conflicting sources), the NLI entailment is not backed by matching numbers,
  or NLI probabilities are inconclusive (between uncertain_threshold and the
  decision thresholds).
* PARTIALLY_SUPPORTED - the full claim is not entailed, but a component is
  (e.g. the effect of "A because B" is supported while the causal link is not).
* INSUFFICIENT_EVIDENCE - no relevant passage entails or contradicts the claim.

"Relevant" means cosine similarity between claim and passage >= relevance_threshold.
These labels describe the relationship between a claim and the *provided
sources*; they do not establish real-world truth.
"""

from __future__ import annotations

import logging

from .claims import decompose_causal, extract_claims
from .config import VerificationConfig
from .index import EvidenceIndex
from .models import NLIModel
from .retrieval import AdaptiveRetriever
from .schema import ClaimStatus, ClaimVerification, EvidenceJudgement, ScoredEvidence
from .text_utils import extract_numbers

logger = logging.getLogger(__name__)


class ClaimVerifier:
    def __init__(self, index: EvidenceIndex, retriever: AdaptiveRetriever, nli: NLIModel,
                 config: VerificationConfig | None = None):
        self.index = index
        self.retriever = retriever
        self.nli = nli
        self.config = config or VerificationConfig()
        self.nli_checks = 0   # number of (premise, claim) pairs scored by the NLI model

    # -- evidence gathering ----------------------------------------------------------
    def candidate_premises(self, claim: str, pool: list[ScoredEvidence] | None = None
                           ) -> list[tuple[str, list[str], float]]:
        """Candidate evidence for a claim: single sentences and two-sentence windows, sorted
        by relevance. Returns (premise_text, evidence_ids, relevance) tuples."""
        cfg = self.config
        ranked = self.retriever.rank(claim, cfg.evidence_per_claim, mode="hybrid")
        by_id = {e.unit.evidence_id: e for e in ranked}
        for item in pool or []:
            by_id.setdefault(item.unit.evidence_id, item)
        if not by_id:
            return []
        claim_vec = self.index.embedder.encode([claim])[0]
        id_to_index = {u.evidence_id: i for i, u in enumerate(self.index.units)}
        indices = [id_to_index[eid] for eid in by_id if eid in id_to_index]
        relevance = {i: float(self.index.embeddings[i] @ claim_vec) for i in indices}
        indices.sort(key=lambda i: -relevance[i])
        indices = indices[: cfg.evidence_per_claim]

        premises = [(self.index.units[i].text, [self.index.units[i].evidence_id], relevance[i]) for i in indices]
        if cfg.use_windows:
            # Evidence split across adjacent sentences ("X happened. This was because Y.").
            windows = []
            for i in indices[:3]:
                for j in self.index.neighbors(i):
                    a, b = sorted((i, j))
                    ids = [self.index.units[a].evidence_id, self.index.units[b].evidence_id]
                    if all(w[1] != ids for w in windows):
                        windows.append((f"{self.index.units[a].text} {self.index.units[b].text}", ids))
            if windows:
                window_vecs = self.index.embedder.encode([w[0] for w in windows])
                premises += [(text, ids, float(vec @ claim_vec)) for (text, ids), vec in zip(windows, window_vecs)]
        if cfg.use_multi_sentence and len(indices) >= 3:
            # Claims that combine facts from non-adjacent sentences (typical of summaries):
            # the top-k sentences together, in document order. Used only as support.
            top = sorted(indices[: cfg.multi_sentence_k])
            text = " ".join(self.index.units[i].text for i in top)
            vec = self.index.embedder.encode([text])[0]
            premises.append((text, [self.index.units[i].evidence_id for i in top], float(vec @ claim_vec)))
        # Most relevant evidence first (windows included), so budgeted verification
        # checks the passages most likely to decide the claim before the rest.
        premises.sort(key=lambda p: -p[2])
        return premises

    def judge(self, claim: str, premises) -> list[EvidenceJudgement]:
        """Score premises against the claim with the NLI model."""
        self.nli_checks += len(premises)
        probs = self.nli.predict([(premise, claim) for premise, _, _ in premises])
        return [
            EvidenceJudgement(evidence_ids=ids, premise=premise,
                              entailment=round(p["entailment"], 4),
                              contradiction=round(p["contradiction"], 4),
                              neutral=round(p["neutral"], 4), relevance=round(rel, 4))
            for (premise, ids, rel), p in zip(premises, probs)
        ]

    # -- decision -------------------------------------------------------------------------
    def verify(self, claim: str, pool: list[ScoredEvidence] | None = None,
               decompose: bool = True, max_premises: int | None = None) -> ClaimVerification:
        premises = self.candidate_premises(claim, pool)[:max_premises]
        if not premises:
            return ClaimVerification(claim, ClaimStatus.INSUFFICIENT_EVIDENCE,
                                     "No evidence is available in the ingested sources.")
        result = self.decide(claim, self.judge(claim, premises), pool, decompose)
        result.checks_used += len(premises)
        return result

    def decide(self, claim: str, judgements: list[EvidenceJudgement],
               pool: list[ScoredEvidence] | None = None, decompose: bool = True,
               part_premises: int | None = None) -> ClaimVerification:
        """Apply the labelling rules (module docstring) to NLI judgements already collected.

        ``part_premises`` caps the evidence checked per component of a causal claim
        (used by the budgeted verifier); None means no cap.
        """
        cfg = self.config
        if not judgements:
            return ClaimVerification(claim, ClaimStatus.INSUFFICIENT_EVIDENCE,
                                     "No evidence was checked for this claim.")
        relevant = [j for j in judgements if j.relevance >= cfg.relevance_threshold]
        claim_numbers = extract_numbers(claim)

        def numbers_ok(j: EvidenceJudgement) -> bool:
            return claim_numbers <= extract_numbers(j.premise)

        entailing = sorted([j for j in relevant if j.entailment >= cfg.support_threshold],
                           key=lambda j: (-j.entailment, len(j.evidence_ids)))
        supporting = [j for j in entailing if numbers_ok(j)]
        # The small NLI model produces spurious contradictions on two-sentence windows
        # that mention several entities, so a window may contradict only when it is
        # more relevant to the claim than every single sentence.
        best_single = max((j.relevance for j in judgements if len(j.evidence_ids) == 1), default=0.0)
        contradicting = sorted([j for j in relevant
                                if (len(j.evidence_ids) == 1
                                    or (len(j.evidence_ids) == 2 and j.relevance > best_single))
                                and j.relevance >= cfg.contradiction_relevance_threshold
                                and j.contradiction >= cfg.contradiction_threshold],
                               key=lambda j: (-j.contradiction, len(j.evidence_ids)))
        # Report single-sentence evidence in preference to windows that contain it.
        supporting = _prefer_single(supporting)
        contradicting = _prefer_single(contradicting)

        result = ClaimVerification(
            claim=claim, status=ClaimStatus.INSUFFICIENT_EVIDENCE, explanation="",
            supporting=supporting[:3], contradicting=contradicting[:3],
            best_entailment=max((j.entailment for j in relevant), default=0.0),
            best_contradiction=max((j.contradiction for j in relevant), default=0.0),
            best_relevance=max(j.relevance for j in judgements),
        )

        if supporting and contradicting:
            # Evidence strength = relevance x NLI probability. When one side is clearly
            # stronger it decides; otherwise the sources genuinely conflict (UNCERTAIN below).
            support_strength = max(j.relevance * j.entailment for j in supporting)
            contra_strength = max(j.relevance * j.contradiction for j in contradicting)
            if contra_strength < support_strength - cfg.conflict_margin:
                contradicting = result.contradicting = []
            elif support_strength < contra_strength - cfg.conflict_margin:
                supporting = result.supporting = []
        if supporting and contradicting:
            result.status = ClaimStatus.UNCERTAIN
            result.explanation = ("The sources conflict: some evidence supports the claim and other "
                                  "evidence contradicts it.")
            return result
        if supporting:
            result.status = ClaimStatus.SUPPORTED
            result.explanation = f"Entailed by evidence (P(entailment)={supporting[0].entailment:.2f})."
            return result
        if contradicting:
            result.status = ClaimStatus.CONTRADICTED
            result.explanation = (f"Relevant evidence contradicts the claim "
                                  f"(P(contradiction)={contradicting[0].contradiction:.2f}).")
            return result
        if entailing:
            result.status = ClaimStatus.UNCERTAIN
            result.supporting = entailing[:3]
            result.explanation = ("The NLI model found entailing evidence, but the numbers in the claim "
                                  "do not all appear in it.")
            return result

        parts = decompose_causal(claim) if decompose and part_premises != 0 else None
        if parts is not None:
            sub_claims = [parts.effect] + ([parts.cause] if parts.cause else [])
            result.parts = [self.verify(c, pool, decompose=False, max_premises=part_premises)
                            for c in sub_claims]
            result.checks_used += sum(p.checks_used for p in result.parts)
            statuses = [p.status for p in result.parts]
            if ClaimStatus.CONTRADICTED in statuses:
                result.status = ClaimStatus.CONTRADICTED
                result.contradicting = [j for p in result.parts for j in p.contradicting][:3]
                result.explanation = "Part of the claim is contradicted by the evidence."
                return result
            if ClaimStatus.SUPPORTED in statuses:
                result.status = ClaimStatus.PARTIALLY_SUPPORTED
                result.supporting = [j for p in result.parts for j in p.supporting][:3]
                established = [p.claim for p in result.parts if p.status == ClaimStatus.SUPPORTED]
                result.explanation = (
                    f"Only part of the claim is established ({'; '.join(established)}). "
                    f"The '{parts.connective}' relationship itself is not supported by the evidence.")
                return result

        if not relevant:
            result.explanation = "No sufficiently relevant evidence was found in the sources."
        elif max(result.best_entailment, result.best_contradiction) >= cfg.uncertain_threshold:
            result.status = ClaimStatus.UNCERTAIN
            result.explanation = ("Relevant evidence exists but the NLI model is inconclusive "
                                  f"(max P(entailment)={result.best_entailment:.2f}, "
                                  f"max P(contradiction)={result.best_contradiction:.2f}).")
        else:
            result.explanation = ("Relevant passages were found, but none of them establishes or "
                                  "contradicts the claim.")
        return result

    def verify_text(self, text: str, pool: list[ScoredEvidence] | None = None) -> list[ClaimVerification]:
        """Extract claims from arbitrary text (e.g. an LLM answer) and verify each one."""
        return [self.verify(claim, pool) for claim in extract_claims(text)]


def _prefer_single(judgements: list[EvidenceJudgement]) -> list[EvidenceJudgement]:
    singles = {j.evidence_ids[0] for j in judgements if len(j.evidence_ids) == 1}
    return [j for j in judgements
            if len(j.evidence_ids) == 1 or not any(eid in singles for eid in j.evidence_ids)]
