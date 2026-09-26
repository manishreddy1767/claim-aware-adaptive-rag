"""Hallucination prevention for arbitrary answer text.

Given a draft answer (e.g. from a chatbot or a local LLM) and the verification
result of each of its claims, build a revised answer that states only what
the sources establish:

* SUPPORTED            -> kept, with a citation to the supporting sentence.
* PARTIALLY_SUPPORTED  -> only the established part is kept, with a note that
                          the rest (e.g. a causal link) is not established.
* UNCERTAIN            -> kept but explicitly marked as unverified.
* CONTRADICTED         -> removed; replaced by a correction quoting and
                          citing the contradicting source sentence.
* INSUFFICIENT_EVIDENCE-> removed and listed as not found in the sources.

This prevents unsupported statements from being presented as established
facts. It cannot guarantee factual truth: it only enforces agreement with the
provided sources, as judged by the NLI model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .index import EvidenceIndex
from .schema import ClaimStatus, ClaimVerification, EvidenceUnit


@dataclass
class RevisedAnswer:
    text: str
    abstained: bool
    kept: list[str] = field(default_factory=list)
    qualified: list[str] = field(default_factory=list)
    corrected: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    citations: list[EvidenceUnit] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"text": self.text, "abstained": self.abstained, "kept": self.kept,
                "qualified": self.qualified, "corrected": self.corrected, "removed": self.removed,
                "citations": [{"marker": i + 1, **u.to_dict(), "citation": u.citation()}
                              for i, u in enumerate(self.citations)]}


def revise_answer(verifications: list[ClaimVerification], index: EvidenceIndex,
                  abstain_message: str) -> RevisedAnswer:
    units = {u.evidence_id: u for u in index.units}
    revised = RevisedAnswer(text="", abstained=False)

    def cite(evidence_ids: list[str]) -> str:
        markers = []
        for eid in evidence_ids:
            unit = units.get(eid)
            if unit is None:
                continue
            if unit not in revised.citations:
                revised.citations.append(unit)
            markers.append(f"[{revised.citations.index(unit) + 1}]")
        return "".join(markers)

    sentences: list[str] = []
    for v in verifications:
        if v.status == ClaimStatus.SUPPORTED and v.supporting:
            sentences.append(f"{v.claim} {cite(v.supporting[0].evidence_ids)}")
            revised.kept.append(v.claim)
        elif v.status == ClaimStatus.PARTIALLY_SUPPORTED:
            established = [p for p in v.parts if p.status == ClaimStatus.SUPPORTED and p.supporting]
            for part in established:
                sentences.append(f"{part.claim} {cite(part.supporting[0].evidence_ids)}")
            sentences.append(f"(The sources do not establish the rest of the statement: \"{v.claim}\")")
            revised.qualified.append(v.claim)
        elif v.status == ClaimStatus.UNCERTAIN:
            sentences.append(f"[Unverified] {v.claim}")
            revised.qualified.append(v.claim)
        elif v.status == ClaimStatus.CONTRADICTED and v.contradicting:
            evidence = v.contradicting[0]
            sentences.append(f"Correction: the claim \"{v.claim}\" conflicts with the sources, which state: "
                             f"{evidence.premise} {cite(evidence.evidence_ids)}")
            revised.corrected.append(v.claim)
        else:
            revised.removed.append(v.claim)

    if not revised.kept and not revised.corrected and not any(
            v.status == ClaimStatus.PARTIALLY_SUPPORTED for v in verifications):
        revised.abstained = True
        revised.text = abstain_message
        if revised.qualified:
            revised.text += " Unverified statements: " + " ".join(sentences)
    else:
        revised.text = " ".join(sentences)
    if revised.removed:
        revised.text += (" Removed because the sources do not support them: "
                         + "; ".join(f"\"{c}\"" for c in revised.removed) + ".")
    return revised
