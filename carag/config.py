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
    # nli-deberta-v3-base measured better than -small on SciFact (macro-F1 0.649 vs 0.590)
    # and on the synthetic claims; -small remains available for low-memory machines.
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    # "auto" picks CUDA when available and falls back to CPU.
    device: str = "auto"
    batch_size: int = 32


@dataclass
class IngestionConfig:
    min_sentence_words: int = 3
    max_sentence_chars: int = 1200
    request_timeout: int = 20
    max_download_bytes: int = 10_000_000
    # OCR pages without a text layer (scanned PDFs) when pypdfium2 + rapidocr are installed.
    ocr: bool = True
    ocr_scale: float = 3.0
    # Render pages whose static HTML yields fewer than js_min_units sentences in a
    # headless browser (optional dependency: playwright + its chromium).
    render_js: bool = True
    js_min_units: int = 3


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
    # Expansion step (a): search again for query key terms missing from the evidence.
    expand_missing_terms: bool = True
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
    # Split "A because B" claims into components when the full claim is not settled.
    decompose_causal: bool = True
    # Also test the top-k most relevant sentences together as one supporting premise.
    use_multi_sentence: bool = True
    multi_sentence_k: int = 3


@dataclass
class BudgetConfig:
    """Shared evidence-check budget for verifying several claims (see carag/budget.py)."""

    enabled: bool = True
    # Total budget = checks_per_claim x number of claims (NLI premise checks).
    checks_per_claim: float = 4.0
    # Floor on the total budget so answers with one or two claims are not starved.
    min_budget: int = 6
    # Premises checked per retrieval step, and the per-claim cap.
    step_size: int = 2
    max_checks_per_claim: int = 12
    # Priority = gap_weight * evidence gap + (1 - gap_weight) * linguistic uncertainty.
    gap_weight: float = 0.7
    # Evidence gain below this counts as a low-gain step (priority is penalized).
    low_gain_threshold: float = 0.05
    # Scheduling penalties: priority / (1 + a * attempts) / (1 + b * low_gain_streak).
    attempt_penalty: float = 0.8
    low_gain_penalty: float = 0.75
    # "priority" (evidence-gain aware) or "round_robin" (baseline).
    strategy: str = "priority"
    # Answer-level accounting: one adaptive-retrieval expansion round costs this many
    # evidence checks. With answer_budget unset, the cost of the rounds actually run is
    # added on top of the verification budget (reported, not limited). With answer_budget
    # set, it is a hard total: expansion rounds are capped so min_budget is left for
    # verification, and verification gets the remainder.
    retrieval_round_cost: int = 2
    answer_budget: int | None = None


@dataclass
class AnswerConfig:
    max_answer_sentences: int = 3
    # Only answer when the top answer sentence has at least this hybrid score.
    min_answer_score: float = 0.40
    check_question_premise: bool = True
    # Multi-part questions (procedures, lists, "including X and Y"): also take retrieved
    # sentences from the anchor's document scoring >= multi_part_relative x the best
    # there, and the rest of their sections when a section has <= max_section_units.
    multi_part_sentences: int = 6
    multi_part_relative: float = 0.75
    max_section_units: int = 6
    # Answer relevance / answerability via an extractive QA model (carag/relevance.py).
    relevance_check: bool = True
    relevance_model: str = "deepset/minilm-uncased-squad2"
    # Abstain when the QA model's best-span margin over "no answer" is below this
    # (0 = the model's own no-answer decision; not tuned).
    answerability_margin: float = 0.0
    relevance_context_k: int = 8
    # When the QA model finds no answer, say "not specified" (topic covered, detail missing)
    # instead of the generic abstention if the best evidence has at least this cosine
    # similarity to the question. Set from 13 handbook/synthetic questions: covered topics
    # scored 0.53-0.73, absent topics 0.26-0.41. Heuristic, not calibrated.
    not_specified_min_semantic: float = 0.50
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
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    answer: AnswerConfig = field(default_factory=AnswerConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def override(self, assignment: str) -> None:
        """Apply 'section.field=value' (e.g. 'verification.use_windows=false')."""
        path, _, raw = assignment.partition("=")
        section, _, name = path.strip().partition(".")
        target = getattr(self, section)
        current = getattr(target, name)
        if isinstance(current, bool):
            value = raw.strip().lower() in ("1", "true", "yes", "on")
        else:
            value = type(current)(raw.strip())
        setattr(target, name, value)
