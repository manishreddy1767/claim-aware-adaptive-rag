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


def test_rejected_premise_includes_section_context(handbook):
    result = handbook.ask("Does the company guarantee 30 days of paid annual leave to every employee?")
    assert result.answer.startswith("No.")
    assert "24 days" in result.answer and "prorated" in result.answer


def test_missing_detail_on_covered_topic_says_not_specified(handbook):
    result = handbook.ask("What is the company's maternity leave duration in weeks?")
    assert result.abstained   # still no answer to the question asked
    assert "do not specify" in result.answer and "maternity leave" in result.answer


@pytest.mark.parametrize("question, facts", [
    ("What is the procedure for requesting work-from-home permission, including the deadline and approval "
     "requirements?", ["PeoplePortal", "3:00 p.m.", "the date, the work location", "manager", "not automatic"]),
    ("What employee benefits are available, and what are the annual reimbursement limits?",
     ["20,000", "5,000", "health insurance", "retirement", "Prior manager approval"]),
    ("Can employees carry forward unused leave? Explain the limits and expiration rules.",
     ["Up to 5", "March 31", "expire", "Personal leave cannot"]),
])
def test_multi_part_answers_are_complete(handbook, question, facts):
    answer = handbook.ask(question).answer
    assert all(f in answer for f in facts), [f for f in facts if f not in answer]
