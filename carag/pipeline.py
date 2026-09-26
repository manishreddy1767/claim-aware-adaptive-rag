"""High-level facade used by the CLI, the web UI and the evaluation scripts."""

from __future__ import annotations

import logging

from .answering import AnswerResult, GroundedAnswerer
from .config import RAGConfig
from .index import EvidenceIndex
from .ingestion import load_bytes, load_source
from .models import get_embedder, get_nli
from .retrieval import AdaptiveRetriever, RetrievalResult
from .schema import ClaimVerification, SourceDocument
from .verification import ClaimVerifier

logger = logging.getLogger(__name__)


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
        self._answerer: GroundedAnswerer | None = None

    # -- components ----------------------------------------------------------------
    @property
    def verifier(self) -> ClaimVerifier:
        if self._verifier is None:
            self._verifier = ClaimVerifier(self.index, self.retriever, get_nli(self.config.models),
                                           self.config.verification)
        return self._verifier

    @property
    def answerer(self) -> GroundedAnswerer:
        if self._answerer is None:
            self._answerer = GroundedAnswerer(self.retriever, self.verifier, self.config.answer)
        return self._answerer

    def apply_config(self, config: RAGConfig) -> None:
        """Swap thresholds without reloading models or re-indexing."""
        self.config = config
        self.retriever.config = config.retrieval
        self.retriever.scorer.config = config.retrieval
        if self._verifier is not None:
            self._verifier.config = config.verification
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
        return self.verifier.verify_text(text)
