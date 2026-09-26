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
