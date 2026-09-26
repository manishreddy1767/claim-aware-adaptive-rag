from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TEST_PDF = ROOT / "claim_aware_rag_test_document.pdf"


@pytest.fixture(scope="session")
def test_pdf() -> Path:
    if not TEST_PDF.exists():
        from scripts.make_test_document import build
        build(TEST_PDF)
    return TEST_PDF


@pytest.fixture(scope="session")
def rag(test_pdf):
    """Pipeline with the synthetic test document ingested (loads models once)."""
    try:
        from carag.pipeline import ClaimAwareRAG
        system = ClaimAwareRAG()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Models unavailable: {exc}")
    system.add_source(str(test_pdf))
    return system


def unit_containing(rag, substring: str):
    return next(u for u in rag.index.units if substring.lower() in u.text.lower())
