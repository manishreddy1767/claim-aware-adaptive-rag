"""Answer relevance: supported evidence that does not answer the question (report Test 1)."""

from __future__ import annotations

import pytest

# The Test 1 failure: a book passage that lists other heroines and never mentions the one asked about.
BOOK = """Heroines of the Novel

Chapter One
The heroines discussed in this volume are Diana Vernon, Argemone Lavington, Beatrix Esmond, and Barbara Grant. Diana Vernon is bold, witty and fiercely independent. Argemone Lavington is proud and intellectual, yet capable of deep devotion. Beatrix Esmond is beautiful, ambitious and restless. Barbara Grant is quick-tongued, playful and loyal to her friends. Each of these women shaped the reader's idea of a heroine in her own century.
"""


@pytest.fixture(scope="module")
def book():
    try:
        from carag.pipeline import ClaimAwareRAG
        system = ClaimAwareRAG()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Models unavailable: {exc}")
    system.add_bytes(BOOK.encode(), "heroines.txt")
    return system


def test_absent_entity_abstains(book):
    result = book.ask("Who is Elizabeth Bennet, and what are her three main personality traits?")
    assert result.abstained
    assert "do not mention Elizabeth Bennet" in result.answer
    assert not result.claims


def test_present_entity_is_answered(book):
    result = book.ask("What are Diana Vernon's main personality traits?")
    assert not result.abstained and "bold, witty" in result.answer


def test_copied_sentence_support_is_flagged(book):
    result = book.ask("What are Diana Vernon's main personality traits?")
    claim = result.claims[0]
    assert claim.status.value == "SUPPORTED" and claim.self_supported
    assert claim.to_dict()["self_supported"] is True
