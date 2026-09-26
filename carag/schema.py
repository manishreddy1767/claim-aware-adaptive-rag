"""Shared data structures."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum


@dataclass
class EvidenceUnit:
    """One sentence of source text with enough metadata to cite it."""

    evidence_id: str          # e.g. "report.pdf#p2.s5"
    text: str
    source: str               # filename or URL
    source_type: str          # "pdf" | "txt" | "docx" | "url"
    position: int             # sentence index within the source
    page: int | None = None   # 1-based page number for PDFs
    url: str | None = None
    section: str | None = None

    def citation(self) -> str:
        where = self.url or self.source
        if self.page is not None:
            return f"{where}, page {self.page}"
        return where

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SourceDocument:
    source: str
    source_type: str
    units: list[EvidenceUnit]
    pages: int | None = None
    title: str | None = None
    warnings: list[str] = field(default_factory=list)
    dropped_units: int = 0


class ClaimStatus(str, Enum):
    """Verification labels. See README 'Claim verification' for definitions."""

    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class ScoredEvidence:
    unit: EvidenceUnit
    score: float
    semantic: float
    lexical: float
    coverage: float
    bonus: float = 0.0
    found_by: str = "initial"   # which retrieval round/action surfaced it

    def to_dict(self) -> dict:
        data = asdict(self)
        data["unit"] = self.unit.to_dict()
        return data


@dataclass
class EvidenceJudgement:
    """NLI judgement of one premise (evidence) against one claim."""

    evidence_ids: list[str]
    premise: str
    entailment: float
    contradiction: float
    neutral: float
    relevance: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ClaimVerification:
    claim: str
    status: ClaimStatus
    explanation: str
    supporting: list[EvidenceJudgement] = field(default_factory=list)
    contradicting: list[EvidenceJudgement] = field(default_factory=list)
    parts: list["ClaimVerification"] = field(default_factory=list)
    best_entailment: float = 0.0
    best_contradiction: float = 0.0
    best_relevance: float = 0.0
    checks_used: int = 0          # NLI evidence checks spent on this claim
    priority: float | None = None  # scheduling priority when verified under a budget

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "status": self.status.value,
            "explanation": self.explanation,
            "supporting": [j.to_dict() for j in self.supporting],
            "contradicting": [j.to_dict() for j in self.contradicting],
            "parts": [p.to_dict() for p in self.parts],
            "best_entailment": self.best_entailment,
            "best_contradiction": self.best_contradiction,
            "best_relevance": self.best_relevance,
            "checks_used": self.checks_used,
            "priority": self.priority,
        }
