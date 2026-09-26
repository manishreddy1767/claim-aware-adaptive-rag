"""Centralized configuration for the Claim-Aware Adaptive RAG pipeline.

All thresholds live here so the CLI, the Streamlit UI and the evaluation
scripts share one set of defaults. Values are heuristics chosen on the
synthetic development set (see evaluation/README.md); they are not
calibrated probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class ModelConfig:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    # "auto" picks CUDA when available and falls back to CPU.
    device: str = "auto"
    batch_size: int = 32


@dataclass
class IngestionConfig:
    min_sentence_words: int = 3
    max_sentence_chars: int = 1200
    request_timeout: int = 20
    max_download_bytes: int = 10_000_000


@dataclass
class RetrievalConfig:
    # Hybrid score = semantic_weight * cosine + lexical_weight * BM25(normalized)
    #                + intent bonuses (only when the sentence shares query terms).
    semantic_weight: float = 0.65
    lexical_weight: float = 0.35
    intent_bonus: float = 0.08
    # Adaptive selection.
    min_k: int = 2
    initial_k: int = 4
    max_k: int = 10
    # Keep candidates scoring at least this fraction of the best score.
    relative_cutoff: float = 0.70
    absolute_min_score: float = 0.25
    # Sufficiency: best hybrid score and query key-term coverage.
    sufficiency_score: float = 0.45
    sufficiency_coverage: float = 0.60
    # Score-bar reduction when every query key term appears in the evidence.
    full_coverage_relief: float = 0.10
    max_expansion_rounds: int = 2
    # Add sentences adjacent to top hits during expansion (disable when units
    # are independent documents rather than consecutive sentences).
    expand_neighbors: bool = True
    # Near-duplicate suppression (cosine between evidence sentences).
    redundancy_threshold: float = 0.92
    # Fixed top-k used by the baseline retriever.
    baseline_k: int = 5


@dataclass
class VerificationConfig:
    # Evidence candidates checked per claim.
    evidence_per_claim: int = 6
    # Minimum semantic similarity for evidence to count as "about" the claim.
    relevance_threshold: float = 0.40
    support_threshold: float = 0.70
    contradiction_threshold: float = 0.70
    # Contradicting evidence must be about the same thing as the claim, so it
    # needs higher similarity than evidence merely considered "relevant".
    contradiction_relevance_threshold: float = 0.55
    # Support and contradiction count as a conflict only if their strength
    # (relevance x NLI probability) is within this margin; otherwise the
    # stronger evidence decides.
    conflict_margin: float = 0.10
    # NLI probability range treated as inconclusive (UNCERTAIN).
    uncertain_threshold: float = 0.40
    # Also test adjacent-sentence windows as premises.
    use_windows: bool = True


@dataclass
class AnswerConfig:
    max_answer_sentences: int = 3
    # Only answer when the top answer sentence has at least this hybrid score.
    min_answer_score: float = 0.40
    check_question_premise: bool = True
    abstain_message: str = (
        "I could not find sufficient evidence in the provided sources to "
        "answer this question reliably."
    )


@dataclass
class RAGConfig:
    models: ModelConfig = field(default_factory=ModelConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    answer: AnswerConfig = field(default_factory=AnswerConfig)

    def to_dict(self) -> dict:
        return asdict(self)
