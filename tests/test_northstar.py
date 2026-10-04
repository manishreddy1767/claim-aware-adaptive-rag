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


def test_comparison_answers_each_side(handbook):
    result = handbook.ask("Compare the company's annual leave policy with its work-from-home policy, "
                          "including eligibility, approval procedures, and restrictions.")
    assert not result.abstained
    leave, _, wfh = result.answer.partition("Work-from-home policy:")
    assert "24 days" in leave and "5 working days" in leave
    assert "probation" in wfh and "3:00 p.m." in wfh


def test_comparison_reports_missing_side(handbook):
    result = handbook.ask("Compare the annual leave policy with the travel reimbursement policy.")
    assert "Travel reimbursement policy: the sources do not describe this." in result.answer
    assert handbook.ask("Compare the stock option plan with the company car policy.").abstained


def test_verifier_labels_undocumented_claim_not_specified(handbook):
    from carag.schema import ClaimStatus
    verdict = handbook.verify_claim("The company provides paid leave for pet adoption.")
    assert verdict.status == ClaimStatus.NOT_SPECIFIED
    # A real contradiction stays CONTRADICTED.
    assert handbook.verify_claim("Full-time employees receive 30 days of paid leave per calendar year.").status \
        == ClaimStatus.CONTRADICTED


def test_checked_answer_does_not_call_undocumented_claim_wrong(handbook):
    check = handbook.check_answer("Employees get paid leave for pet adoption.")
    assert "Not established" in check.revised.text and "Correction" not in check.revised.text

