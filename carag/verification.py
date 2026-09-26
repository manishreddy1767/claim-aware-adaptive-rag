"""Claim-level verification against ingested evidence using an NLI cross-encoder.

Labels (applied in this order):

* SUPPORTED - a topically relevant evidence passage entails the full claim
  (P(entailment) >= support_threshold) and every number in the claim appears in
  that passage, and no relevant passage contradicts it.
* CONTRADICTED - a topically relevant passage contradicts the claim
  (P(contradiction) >= contradiction_threshold) and none supports it, or a
  component of a causal claim is contradicted.
* UNCERTAIN - relevant passages both support and contradict the claim
  (conflicting sources), the NLI entailment is not backed by matching numbers,
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

    # -- evidence gathering ----------------------------------------------------------
    def _premises(self, claim: str, pool: list[ScoredEvidence] | None) -> list[tuple[str, list[str], float]]:
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
            # Evidence split across adjacent sentences ("X happened. This was because Y.")
            for i in indices[:3]:
                for j in self.index.neighbors(i):
                    a, b = sorted((i, j))
                    ids = [self.index.units[a].evidence_id, self.index.units[b].evidence_id]
                    if any(p[1] == ids for p in premises):
                        continue
                    rel = max(relevance.get(a, 0.0), relevance.get(b, 0.0),
                              float(self.index.embeddings[j] @ claim_vec))
                    premises.append((f"{self.index.units[a].text} {self.index.units[b].text}", ids, rel))
        return premises

    def _judge(self, claim: str, premises) -> list[EvidenceJudgement]:
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
               decompose: bool = True) -> ClaimVerification:
        cfg = self.config
        premises = self._premises(claim, pool)
        if not premises:
            return ClaimVerification(claim, ClaimStatus.INSUFFICIENT_EVIDENCE,
                                     "No evidence is available in the ingested sources.")
        judgements = self._judge(claim, premises)
        relevant = [j for j in judgements if j.relevance >= cfg.relevance_threshold]
        claim_numbers = extract_numbers(claim)

        def numbers_ok(j: EvidenceJudgement) -> bool:
            return claim_numbers <= extract_numbers(j.premise)

        entailing = sorted([j for j in relevant if j.entailment >= cfg.support_threshold],
                           key=lambda j: (-j.entailment, len(j.evidence_ids)))
        supporting = [j for j in entailing if numbers_ok(j)]
        contradicting = sorted([j for j in relevant if j.contradiction >= cfg.contradiction_threshold],
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

        parts = decompose_causal(claim) if decompose else None
        if parts is not None:
            sub_claims = [parts.effect] + ([parts.cause] if parts.cause else [])
            result.parts = [self.verify(c, pool, decompose=False) for c in sub_claims]
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
