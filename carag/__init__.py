"""Claim-Aware Adaptive RAG: adaptive retrieval + claim-level verification."""

from .config import RAGConfig
from .schema import ClaimStatus, EvidenceUnit, SourceDocument

__all__ = ["RAGConfig", "ClaimStatus", "EvidenceUnit", "SourceDocument", "ClaimAwareRAG"]


def __getattr__(name):
    # Import lazily so `import carag` does not pull in torch.
    if name == "ClaimAwareRAG":
        from .pipeline import ClaimAwareRAG
        return ClaimAwareRAG
    raise AttributeError(name)
