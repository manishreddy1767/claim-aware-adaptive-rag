"""Grounded extractive answering with claim verification and abstention.

The answer is composed only of sentences copied from retrieved evidence, so
every answer sentence has a real citation. Claim verification then checks each
answer claim against *all* relevant evidence, which catches conflicting sources
and qualifies or removes claims that are not established. For yes/no and "why"
questions the question's own premise is verified, so misleading assumptions are
reported rather than answered.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .claims import extract_claims
from .config import AnswerConfig
from .retrieval import AdaptiveRetriever, RetrievalResult
from .schema import ClaimStatus, ClaimVerification, EvidenceUnit, ScoredEvidence
from .text_utils import contains_marker
from .budget import BudgetedVerifier, BudgetReport
from .verification import ClaimVerifier

_CAVEAT_MARKERS = ("however", "limitation", "limitations", "caveat", "preliminary",
                   "should be interpreted", "small sample", "not statistically", "cannot be",
                   "unclear", "was not measured", "were not measured", "not controlled")

# Evidence saying something is undocumented ("no pet-adoption benefit is established",
# "the duration is not specified") rather than that it is false.
_ABSENCE = re.compile(
    r"\b(?:not|no)\b[^.;]{0,60}?\b(?:established|specified|stated|mentioned|covered|documented|defined)\b|"
    r"\b(?:does|do|did) not (?:specify|state|mention|cover|establish|define)\b", re.I)

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
    kind: str = "answer"           # "answer" | "caveat" | "correction" | "context"
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
    budget: BudgetReport | None = None
    answer_span: str | None = None          # short answer extracted by the QA model
    answerability: float | None = None      # QA margin: best span score - "no answer" score

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
            "budget": self.budget.to_dict() if self.budget else None,
            "answer_span": self.answer_span,
            "answerability": self.answerability,
        }


class GroundedAnswerer:
    def __init__(self, retriever: AdaptiveRetriever, verifier: ClaimVerifier,
                 config: AnswerConfig | None = None, budgeted: BudgetedVerifier | None = None,
                 span_qa=None):
        self.retriever = retriever
        self.verifier = verifier
        self.config = config or AnswerConfig()
        self.budgeted = budgeted   # shared evidence budget across answer claims (optional)
        self.span_qa = span_qa     # extractive QA model for answer relevance (optional)

    def _locate_answer(self, question: str, retrieval: RetrievalResult):
        """Run the QA model over the best evidence; return (span, margin, evidence sentence)."""
        pool = {e.unit.evidence_id: e for e in self.retriever.rank(question, self.config.relevance_context_k)}
        for e in retrieval.evidence:
            pool.setdefault(e.unit.evidence_id, e)
        ordered = sorted(pool.values(), key=lambda e: (e.unit.source, e.unit.position))
        context, spans = "", []
        for e in ordered:
            start = len(context)
            context += e.unit.text + " "
            spans.append((start, start + len(e.unit.text), e))
        span = self.span_qa.answer(question, context)
        holder = next((e for a, b, e in spans if a <= span.start < b), None) if span.text else None
        return span, holder

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

    def _complete_multi_part(self, question: str, retrieval: RetrievalResult, selected: list[ScoredEvidence],
                             anchor: ScoredEvidence) -> list[ScoredEvidence]:
        """Extend the answer to a multi-part question (procedure, list, rules).

        The parts of such an answer are usually spread over one document: several
        retrieved sentences of similar score (one per benefit), and the rest of a short
        section ("... must be used by March 31." before "... not used by that date expire.").
        Every added sentence is still verified like any other answer sentence.
        """
        cfg = self.config
        index = self.retriever.index
        analysis = retrieval.analysis
        same_doc = [e for e in retrieval.evidence if e.unit.source == anchor.unit.source]
        best = max([e.score for e in same_doc] + [anchor.score])
        core = {e.unit.evidence_id: e for e in [anchor] + selected}
        for e in same_doc:
            if e.score >= cfg.multi_part_relative * best:
                core.setdefault(e.unit.evidence_id, e)

        sizes: dict[tuple, int] = {}
        for u in index.units:
            sizes[(u.source, u.section)] = sizes.get((u.source, u.section), 0) + 1
        sections = {(e.unit.source, e.unit.section) for e in core.values()
                    if e.unit.section and sizes[(e.unit.source, e.unit.section)] <= cfg.max_section_units}
        scores = self.retriever.scorer.score(question, analysis.key_terms, analysis.intents)
        completion = [self.retriever.scorer.make_evidence(i, scores, "section completion")
                      for i, u in enumerate(index.units)
                      if (u.source, u.section) in sections and u.evidence_id not in core]

        # Priority: the anchor and its section, then the strongest of the rest.
        anchor_key = (anchor.unit.source, anchor.unit.section)
        rest = sorted([e for e in list(core.values()) + completion if e.unit.evidence_id != anchor.unit.evidence_id],
                      key=lambda e: ((e.unit.source, e.unit.section) != anchor_key, -e.score))
        chosen = [anchor] + rest[: cfg.multi_part_sentences - 1]
        return sorted(chosen, key=lambda e: (e.unit.source, e.unit.position))

    def _section_context(self, retrieval: RetrievalResult, anchor_ids: list[str]) -> list[ScoredEvidence]:
        """Retrieved sentences from the same section as the anchor evidence (e.g. the
        prorated-leave rule next to '24 days per year'), in document order."""
        anchors = [self._unit_for(eid) for eid in anchor_ids]
        keys = {(u.source, u.section) for u in anchors if u.section}
        extra = [e for e in retrieval.evidence
                 if e.unit.evidence_id not in anchor_ids and (e.unit.source, e.unit.section) in keys]
        extra = sorted(extra, key=lambda e: -e.score)[: self.config.max_answer_sentences - 1]
        return sorted(extra, key=lambda e: e.unit.position)

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

    def _not_specified(self, result: AnswerResult, retrieval: RetrievalResult, reason: str) -> AnswerResult:
        """Abstain, but say the topic is covered and show the closest passage: retrieval found
        the subject (e.g. maternity leave), only the requested detail (weeks) is missing."""
        self._abstain(result, reason)
        top = max(retrieval.evidence, key=lambda e: e.score)
        marker = self._cite(top.unit, result.citations)
        result.answer = ("The sources cover this topic but do not specify the requested detail. "
                         f"The most relevant passage is: {top.unit.text} [{marker}]")
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
                if _ABSENCE.search(text):
                    # "No X is established" means undocumented, not denied: NLI scores it as a
                    # contradiction, but the answer must not claim the opposite is true.
                    lead = "The sources do not establish this."
                    result.notes.append("The evidence states that this is not established or specified, "
                                        "which is not the same as the sources ruling it out.")
                elif analysis.is_yes_no:
                    lead = "No."
                else:
                    lead = "The question's assumption is not supported by the sources."
                result.sentences.append(AnswerSentence(text, markers, [premise], kind="correction"))
                result.answer = f"{lead} The sources state: {text} " + "".join(f"[{m}]" for m in markers)
                for e in self._section_context(retrieval, ids):
                    marker = self._cite(e.unit, result.citations)
                    result.sentences.append(AnswerSentence(e.unit.text, [marker], [], kind="context"))
                    result.answer += f" {e.unit.text} [{marker}]"
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

        # 2. Answerability: with the QA model, it decides whether the evidence contains an
        #    answer and which sentence holds it; otherwise retrieval sufficiency decides.
        span_evidence = None
        if adaptive and self.span_qa is not None:
            span, span_evidence = self._locate_answer(question, retrieval)
            result.answerability = round(span.margin, 3)
            if span.margin < cfg.answerability_margin or span_evidence is None:
                timings["verification_s"] = round(time.perf_counter() - start, 3)
                reason = f"The retrieved evidence does not appear to contain an answer (QA answerability {span.margin:.1f})."
                if retrieval.sufficient and retrieval.evidence:
                    return self._not_specified(result, retrieval, reason)
                return self._abstain(result, reason)
            result.answer_span = span.text.strip()
            if not retrieval.sufficient:
                result.notes.append("Retrieval heuristics flagged: " + " ".join(retrieval.reasons))
        elif adaptive and not retrieval.sufficient:
            timings["verification_s"] = round(time.perf_counter() - start, 3)
            return self._abstain(result, "Insufficient evidence: " + " ".join(retrieval.reasons))

        # 3. Compose an extractive answer from the best evidence.
        selected = self._select_sentences(retrieval)
        if span_evidence is not None:
            others = [e for e in selected if e.unit.evidence_id != span_evidence.unit.evidence_id]
            selected = sorted([span_evidence] + others[: cfg.max_answer_sentences - 1],
                              key=lambda e: (e.unit.source, e.unit.position))
        if not selected:
            timings["verification_s"] = round(time.perf_counter() - start, 3)
            return self._abstain(result, "No retrieved sentence was relevant enough to answer.")
        if analysis.multi_part and adaptive:
            selected = self._complete_multi_part(question, retrieval, selected, span_evidence or selected[0])
        used = {e.unit.evidence_id for e in selected}
        drafts = [(e, "answer") for e in selected] + [(e, "caveat") for e in self._caveats(retrieval, used)]

        # 4. Verify each claim (under a shared evidence budget when enabled);
        #    drop unsupported/contradicted claims, qualify partial/uncertain ones.
        draft_claims = [extract_claims(item.unit.text) or [item.unit.text] for item, _ in drafts]
        flat = [c for claims in draft_claims for c in claims]
        if not verify:
            flat_verdicts = []
        elif self.budgeted is not None and self.budgeted.config.enabled:
            flat_verdicts, result.budget = self.budgeted.verify_claims(flat, retrieval.evidence)
        else:
            flat_verdicts = [self.verifier.verify(c, retrieval.evidence) for c in flat]
        cursor = 0
        for (item, kind), claims in zip(drafts, draft_claims):
            verdicts = flat_verdicts[cursor: cursor + len(claims)] if verify else []
            cursor += len(claims)
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
