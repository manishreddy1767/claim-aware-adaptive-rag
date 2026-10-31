"""Regression tests on the demo document examples/aurora_station_manual.pdf."""

from __future__ import annotations

from pathlib import Path

import pytest

from carag.schema import ClaimStatus

MANUAL = Path(__file__).resolve().parents[1] / "examples" / "aurora_station_manual.pdf"


@pytest.fixture(scope="module")
def manual():
    try:
        from carag.pipeline import ClaimAwareRAG
        system = ClaimAwareRAG()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Models unavailable: {exc}")
    system.add_source(str(MANUAL))
    return system


def test_quoted_claim_contradicted_by_a_later_number_is_a_conflict(manual):
    # Section 2 says 24 residents; the 2026 update says 30.
    verdict = manual.verify_claim("The station can accommodate a maximum of 24 residents at any time.")
    assert verdict.status == ClaimStatus.UNCERTAIN and verdict.contradicting


@pytest.mark.parametrize("claim", ["Visiting researchers can stay at Aurora Lake for up to 30 days.",
                                   "Any researcher may fly the station's drones."])
def test_contradiction_spread_over_sentences_is_found(manual, claim):
    assert manual.verify_claim(claim).status == ClaimStatus.CONTRADICTED


def test_stated_false_assumption_is_rejected(manual):
    result = manual.ask("Since boat requests are approved automatically, how early must I submit one?")
    assert result.outcome == "reject_premise"
    assert "48 hours" in result.answer
