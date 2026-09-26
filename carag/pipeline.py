"""High-level facade used by the CLI, the web UI and the evaluation scripts."""

from __future__ import annotations

import logging

from dataclasses import dataclass

from .answering import AnswerResult, GroundedAnswerer
from .budget import BudgetedVerifier, BudgetReport
from .claims import extract_claims
from .config import RAGConfig
from .index import EvidenceIndex
from .ingestion import load_bytes, load_source
from .models import get_embedder, get_nli
from .retrieval import AdaptiveRetriever, RetrievalResult
from .revision import RevisedAnswer, revise_answer
from .schema import ClaimVerification, SourceDocument
from .verification import ClaimVerifier

logger = logging.getLogger(__name__)


@dataclass
class GeneratedAnswer:
    """A local LLM draft and its verified, revised version."""

    question: str
    draft: str
    draft_declined: bool          # the LLM itself said the evidence has no answer
    check: "AnswerCheck | None"   # None when the draft declined to answer
    retrieval: RetrievalResult

    @property
    def final_text(self) -> str:
        return self.check.revised.text if self.check else self.draft

    @property
    def abstained(self) -> bool:
        return self.draft_declined or (self.check is not None and self.check.revised.abstained)

    def to_dict(self) -> dict:
        return {"question": self.question, "draft": self.draft, "draft_declined": self.draft_declined,
                "final": self.final_text, "abstained": self.abstained,
                "check": self.check.to_dict() if self.check else None}


@dataclass
class AnswerCheck:
    """Claim-level verification and revision of an externally produced answer."""

    claims: list[ClaimVerification]
    revised: RevisedAnswer
    budget: BudgetReport | None

    def to_dict(self) -> dict:
        return {"claims": [c.to_dict() for c in self.claims], "revised": self.revised.to_dict(),
                "budget": self.budget.to_dict() if self.budget else None}


class ClaimAwareRAG:
    """Ingest sources, answer questions with citations, and verify claims.

    The NLI model is loaded lazily on the first verification call so that
    ingestion and retrieval work (and can be tested) without it.
    """

    def __init__(self, config: RAGConfig | None = None):
        self.config = config or RAGConfig()
        self.embedder = get_embedder(self.config.models)
        self.index = EvidenceIndex(self.embedder)
        self.retriever = AdaptiveRetriever(self.index, self.config.retrieval)
        self._verifier: ClaimVerifier | None = None
        self._budgeted: BudgetedVerifier | None = None
        self._answerer: GroundedAnswerer | None = None
        self._generator = None

    # -- components ----------------------------------------------------------------
    @property
    def verifier(self) -> ClaimVerifier:
        if self._verifier is None:
            self._verifier = ClaimVerifier(self.index, self.retriever, get_nli(self.config.models),
                                           self.config.verification)
        return self._verifier

    @property
    def budgeted(self) -> BudgetedVerifier:
        if self._budgeted is None:
            self._budgeted = BudgetedVerifier(self.verifier, self.config.budget)
        return self._budgeted

    @property
    def answerer(self) -> GroundedAnswerer:
        if self._answerer is None:
            span_qa = None
            if self.config.answer.relevance_check:
                from .relevance import SpanQA
                span_qa = SpanQA(self.config.answer.relevance_model, self.config.models.device)
            self._answerer = GroundedAnswerer(self.retriever, self.verifier, self.config.answer,
                                              self.budgeted, span_qa)
        return self._answerer

    def apply_config(self, config: RAGConfig) -> None:
        """Swap thresholds without reloading models or re-indexing."""
        if (self._answerer is not None and
                (config.answer.relevance_check, config.answer.relevance_model)
                != (self.config.answer.relevance_check, self.config.answer.relevance_model)):
            self._answerer = None   # rebuilt lazily with/without the QA model
        if config.models.nli_model != self.config.models.nli_model:
            self._verifier = self._budgeted = self._answerer = None   # reload with the new NLI model
        self.config = config
        self.retriever.config = config.retrieval
        self.retriever.scorer.config = config.retrieval
        if self._verifier is not None:
            self._verifier.config = config.verification
        if self._budgeted is not None:
            self._budgeted.config = config.budget
        if self._answerer is not None:
            self._answerer.config = config.answer

    # -- ingestion -----------------------------------------------------------------
    def add_source(self, source: str) -> SourceDocument:
        document = load_source(source, self.config.ingestion)
        self.index.add_document(document)
        return document

    def add_bytes(self, data: bytes, name: str) -> SourceDocument:
        document = load_bytes(data, name, self.config.ingestion)
        self.index.add_document(document)
        return document

    def add_document(self, document: SourceDocument) -> SourceDocument:
        self.index.add_document(document)
        return document

    # -- querying ------------------------------------------------------------------
    def retrieve(self, question: str) -> RetrievalResult:
        return self.retriever.retrieve(question)

    def ask(self, question: str, verify: bool = True, adaptive: bool = True) -> AnswerResult:
        if not question or not question.strip():
            raise ValueError("Please enter a question.")
        return self.answerer.answer(question.strip(), verify=verify, adaptive=adaptive)

    def generate_answer(self, question: str, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
                        gate: bool = True) -> GeneratedAnswer:
        """Draft an answer with a local LLM, then verify and revise it (hallucination filter)."""
        from .generation import LocalGenerator, is_no_answer
        if self._generator is None:
            self._generator = LocalGenerator(model_name, self.config.models.device)
        retrieval = self.retriever.retrieve(question)
        # Same answerability gate as the extractive path: do not generate an answer
        # when the QA model finds no answer in the evidence.
        if gate and self.answerer.span_qa is not None:
            span, holder = self.answerer._locate_answer(question, retrieval)
            if span.margin < self.config.answer.answerability_margin or holder is None:
                return GeneratedAnswer(question, self.config.answer.abstain_message, True, None, retrieval)
        draft = self._generator.generate(question, retrieval.evidence)
        if is_no_answer(draft):
            return GeneratedAnswer(question, draft, True, None, retrieval)
        return GeneratedAnswer(question, draft, False, self.check_answer(draft), retrieval)

    def verify_claim(self, claim: str) -> ClaimVerification:
        return self.verifier.verify(claim)

    def verify_text(self, text: str) -> list[ClaimVerification]:
        """Verify every claim in the text (exhaustively, no budget)."""
        return self.verifier.verify_text(text)

    def check_answer(self, text: str, budget: int | None = None) -> AnswerCheck:
        """Hallucination detection + prevention for any answer text.

        Claims are verified under the shared evidence budget (if enabled) and the
        answer is rewritten so only source-supported statements remain as facts.
        """
        claims = extract_claims(text)
        report = None
        if self.config.budget.enabled:
            verifications, report = self.budgeted.verify_claims(claims, budget=budget)
        else:
            verifications = [self.verifier.verify(c) for c in claims]
        revised = revise_answer(verifications, self.index, self.config.answer.abstain_message)
        return AnswerCheck(verifications, revised, report)
