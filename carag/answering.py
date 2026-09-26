"""Grounded extractive answering with claim verification and abstention.

The answer is composed only of sentences copied from retrieved evidence, so
every answer sentence has a real citation. Claim verification then checks each
answer claim against *all* relevant evidence, which catches conflicting sources
and qualifies or removes claims that are not established. For yes/no and "why"
questions the question's own premise is verified, so misleading assumptions are
reported rather than answered.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .claims import extract_claims
from .config import AnswerConfig
from .retrieval import AdaptiveRetriever, RetrievalResult
from .schema import ClaimStatus, ClaimVerification, EvidenceUnit, ScoredEvidence
from .text_utils import contains_marker
from .verification import ClaimVerifier

_CAVEAT_MARKERS = ("however", "limitation", "limitations", "caveat", "preliminary",
                   "should be interpreted", "small sample", "not statistically", "cannot be",
                   "unclear", "was not measured", "were not measured", "not controlled")

_STATUS_QUALIFIER = {
    ClaimStatus.PARTIALLY_SUPPORTED: "only partly supported by the sources",
    ClaimStatus.UNCERTAIN: "verification was inconclusive",
}


@dataclass
class Citation:
    marker: int
    unit: EvidenceUnit

    def to_dict(self) -> dict:
        return {"marker": self.marker, **self.unit.to_dict(), "citation": self.unit.citation()}


@dataclass
class AnswerSentence:
    text: str
    citation_markers: list[int]
    claims: list[ClaimVerification]
    kind: str = "answer"           # "answer" | "caveat" | "correction"
    qualifier: str | None = None


@dataclass
class AnswerResult:
    question: str
    answer: str
    abstained: bool
    sentences: list[AnswerSentence] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    premise_check: ClaimVerification | None = None
    removed_claims: list[ClaimVerification] = field(default_factory=list)
    retrieval: RetrievalResult | None = None
    notes: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def claims(self) -> list[ClaimVerification]:
        return [c for s in self.sentences for c in s.claims]

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "abstained": self.abstained,
            "sentences": [{"text": s.text, "citations": s.citation_markers, "kind": s.kind,
                           "qualifier": s.qualifier, "claims": [c.to_dict() for c in s.claims]}
                          for s in self.sentences],
            "citations": [c.to_dict() for c in self.citations],
            "premise_check": self.premise_check.to_dict() if self.premise_check else None,
            "removed_claims": [c.to_dict() for c in self.removed_claims],
            "retrieval": self.retrieval.to_dict() if self.retrieval else None,
            "notes": self.notes,
            "timings": self.timings,
        }


class GroundedAnswerer:
    def __init__(self, retriever: AdaptiveRetriever, verifier: ClaimVerifier,
                 config: AnswerConfig | None = None):
        self.retriever = retriever
        self.verifier = verifier
        self.config = config or AnswerConfig()

    # -- helpers -------------------------------------------------------------------
    def _select_sentences(self, retrieval: RetrievalResult) -> list[ScoredEvidence]:
        cfg = self.config
        analysis = retrieval.analysis
        candidates = sorted(retrieval.evidence, key=lambda e: -e.score)
        if not candidates or candidates[0].score < cfg.min_answer_score:
            return []
        features = self.retriever.index.unit_features
        position = {u.evidence_id: i for i, u in enumerate(self.retriever.index.units)}
        chosen = [candidates[0]]
        covered = {t for t in analysis.key_terms if t in self.retriever.index.unit_terms[position[candidates[0].unit.evidence_id]]}
        wanted = {f for f in ("causal", "numeric", "comparison") if f in analysis.intents}
        have = {f for f in wanted if features[position[candidates[0].unit.evidence_id]][f]}
        for item in candidates[1:]:
            if len(chosen) >= cfg.max_answer_sentences:
                break
            i = position[item.unit.evidence_id]
            terms = {t for t in analysis.key_terms if t in self.retriever.index.unit_terms[i]}
            new_terms = terms - covered
            new_types = {f for f in wanted - have if features[i][f]} if item.coverage > 0 else set()
            strong = item.score >= 0.8 * chosen[0].score
            if (strong and new_terms) or new_types or (strong and item.score >= cfg.min_answer_score + 0.2):
                chosen.append(item)
                covered |= terms
                have |= new_types
        # Present in document order for readability.
        return sorted(chosen, key=lambda e: (e.unit.source, e.unit.position))

    def _caveats(self, retrieval: RetrievalResult, used: set[str]) -> list[ScoredEvidence]:
        return [e for e in retrieval.evidence
                if e.unit.evidence_id not in used and e.coverage >= 0.3
                and contains_marker(e.unit.text, _CAVEAT_MARKERS)][:1]

    @staticmethod
    def _cite(unit: EvidenceUnit, citations: list[Citation]) -> int:
        for c in citations:
            if c.unit.evidence_id == unit.evidence_id:
                return c.marker
        citations.append(Citation(len(citations) + 1, unit))
        return len(citations)

    def _abstain(self, result: AnswerResult, reason: str) -> AnswerResult:
        result.abstained = True
        result.answer = self.config.abstain_message
        result.notes.insert(0, reason)
        return result

    def _unit_for(self, evidence_id: str) -> EvidenceUnit:
        return next(u for u in self.retriever.index.units if u.evidence_id == evidence_id)

    # -- main entry point --------------------------------------------------------------
    def answer(self, question: str, verify: bool = True, adaptive: bool = True) -> AnswerResult:
        cfg = self.config
        timings: dict[str, float] = {}
        result = AnswerResult(question=question, answer="", abstained=False, timings=timings)

        start = time.perf_counter()
        if adaptive:
            retrieval = self.retriever.retrieve(question)
        else:
            from .query import analyze_question
            analysis = analyze_question(question)
            evidence = self.retriever.rank(question, self.retriever.config.baseline_k, "semantic")
            retrieval = RetrievalResult(question, analysis, evidence, sufficient=bool(evidence),
                                        trace=[{"round": 0, "action": f"fixed top-{len(evidence)} semantic"}])
        timings["retrieval_s"] = round(time.perf_counter() - start, 3)
        result.retrieval = retrieval
        analysis = retrieval.analysis

        if not len(self.retriever.index):
            return self._abstain(result, "No sources have been ingested yet.")
        if analysis.is_vague and adaptive:
            return self._abstain(result, analysis.vague_reason)

        # 1. Verify the question's premise (misleading assumptions / yes-no questions).
        start = time.perf_counter()
        if verify and cfg.check_question_premise and analysis.premise:
            premise = self.verifier.verify(analysis.premise, retrieval.evidence)
            result.premise_check = premise
            if premise.status == ClaimStatus.CONTRADICTED and premise.contradicting:
                ids = premise.contradicting[0].evidence_ids
                markers = [self._cite(self._unit_for(eid), result.citations) for eid in ids]
                text = " ".join(self._unit_for(eid).text for eid in ids)
                lead = "No." if analysis.is_yes_no else "The question's assumption is not supported by the sources."
                result.sentences.append(AnswerSentence(text, markers, [premise], kind="correction"))
                result.answer = f"{lead} The sources state: {text} " + "".join(f"[{m}]" for m in markers)
                result.notes.append(f"Premise contradicted: \"{analysis.premise}\"")
                timings["verification_s"] = round(time.perf_counter() - start, 3)
                return result
            if analysis.is_yes_no and premise.status == ClaimStatus.SUPPORTED:
                ids = premise.supporting[0].evidence_ids
                markers = [self._cite(self._unit_for(eid), result.citations) for eid in ids]
                text = " ".join(self._unit_for(eid).text for eid in ids)
                result.sentences.append(AnswerSentence(text, markers, [premise]))
                result.answer = f"Yes. {text} " + "".join(f"[{m}]" for m in markers)
                timings["verification_s"] = round(time.perf_counter() - start, 3)
                return result
            if premise.status in (ClaimStatus.INSUFFICIENT_EVIDENCE, ClaimStatus.UNCERTAIN,
                                  ClaimStatus.PARTIALLY_SUPPORTED):
                result.notes.append(
                    f"The sources do not clearly establish the question's premise \"{analysis.premise}\" "
                    f"({premise.status.value}).")
                if analysis.is_yes_no:
                    timings["verification_s"] = round(time.perf_counter() - start, 3)
                    return self._abstain(result, "The sources neither confirm nor contradict this statement.")

        # 2. Abstain when retrieval could not find sufficient evidence.
        if adaptive and not retrieval.sufficient:
            timings["verification_s"] = round(time.perf_counter() - start, 3)
            return self._abstain(result, "Insufficient evidence: " + " ".join(retrieval.reasons))

        # 3. Compose an extractive answer from the best evidence.
        selected = self._select_sentences(retrieval)
        if not selected:
            timings["verification_s"] = round(time.perf_counter() - start, 3)
            return self._abstain(result, "No retrieved sentence was relevant enough to answer.")
        used = {e.unit.evidence_id for e in selected}
        drafts = [(e, "answer") for e in selected] + [(e, "caveat") for e in self._caveats(retrieval, used)]

        # 4. Verify each claim; drop unsupported/contradicted, qualify partial/uncertain.
        for item, kind in drafts:
            claims = extract_claims(item.unit.text) or [item.unit.text]
            verdicts = [self.verifier.verify(c, retrieval.evidence) for c in claims] if verify else []
            bad = [v for v in verdicts if v.status in (ClaimStatus.CONTRADICTED, ClaimStatus.INSUFFICIENT_EVIDENCE)]
            if bad and len(bad) == len(verdicts):
                result.removed_claims.extend(bad)
                continue
            qualifier = next((_STATUS_QUALIFIER[v.status] for v in verdicts if v.status in _STATUS_QUALIFIER), None)
            marker = self._cite(item.unit, result.citations)
            result.sentences.append(AnswerSentence(item.unit.text, [marker], verdicts, kind, qualifier))
            result.removed_claims.extend(bad)
        timings["verification_s"] = round(time.perf_counter() - start, 3)

        if not any(s.kind == "answer" for s in result.sentences):
            result.sentences.clear()
            result.citations.clear()
            return self._abstain(result, "None of the candidate answer claims could be verified against the sources.")

        parts = []
        for s in result.sentences:
            prefix = "Caveat: " if s.kind == "caveat" else ""
            suffix = f" ({s.qualifier})" if s.qualifier else ""
            parts.append(f"{prefix}{s.text}{suffix} " + "".join(f"[{m}]" for m in s.citation_markers))
        result.answer = " ".join(parts).strip()
        return result
