"""Regression tests on the Northstar handbook (evaluation/datasets/northstar)."""

from __future__ import annotations

from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "evaluation" / "datasets" / "northstar"


@pytest.fixture(scope="module")
def handbook():
    try:
        from carag.pipeline import ClaimAwareRAG
        system = ClaimAwareRAG()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Models unavailable: {exc}")
    for path in sorted(DOCS.glob("*.txt")):
        system.add_source(str(path))
    return system


def test_yes_no_with_follow_up_is_answered(handbook):
    result = handbook.ask("Can employees carry forward unused leave? Explain the limits and expiration rules.")
    assert not result.abstained
    assert "Up to 5 unused annual-leave days" in result.answer


def test_undocumented_benefit_is_not_answered_no(handbook):
    result = handbook.ask("Does the company provide paid leave for pet adoption?")
    assert not result.answer.startswith("No.")
    assert result.answer.startswith("The sources do not establish this.")
