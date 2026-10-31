"""Budget-aware verification and hallucination-prevention (answer revision)."""

import pytest

from carag.budget import linguistic_uncertainty
from carag.schema import ClaimStatus as S

DRAFT = ("Trial Two consumed an average of 118 milliwatt-hours per day. Trial One sampled the sensors at 10 Hz. "
         "Trial Two consumed more energy because its nodes were placed in direct sunlight. "
         "The microcontroller was manufactured by Texas Instruments.")


def test_linguistic_uncertainty_counts_hedges():
    assert linguistic_uncertainty("Trial Two used 10 Hz.") == 0.35
    assert linguistic_uncertainty("Trial Two may possibly have used 10 Hz.") == pytest.approx(0.65)


def test_budget_is_never_exceeded_and_matches_exhaustive_labels(rag):
    check = rag.check_answer(DRAFT)
    report = check.budget
    assert report.used_total <= report.budget == 16          # 4 checks x 4 claims
    budgeted = [c.status for c in check.claims]
    exhaustive = [rag.verify_claim(c.claim).status for c in check.claims]
    assert budgeted == exhaustive == [S.SUPPORTED, S.CONTRADICTED, S.PARTIALLY_SUPPORTED,
                                      S.INSUFFICIENT_EVIDENCE]


def test_easy_claims_stop_early_and_budget_goes_to_hard_claims(rag):
    report = rag.check_answer(DRAFT).budget
    by_claim = {c["claim"]: c for c in report.claims}
    easy = by_claim["Trial Two consumed an average of 118 milliwatt-hours per day."]
    hard = by_claim["The microcontroller was manufactured by Texas Instruments."]
    assert easy["attempts"] == 1 and easy["checks"] == 2
    assert hard["checks"] > easy["checks"]


def test_tiny_budget_marks_unchecked_claims_as_not_verified(rag):
    check = rag.check_answer(DRAFT, budget=2)
    assert check.budget.used_total <= 2
    unchecked = [c for c in check.claims if "budget ran out" in c.explanation]
    assert unchecked and all(c.status != S.SUPPORTED for c in unchecked)


def test_round_robin_strategy_runs(rag):
    rag.config.budget.strategy = "round_robin"
    try:
        check = rag.check_answer(DRAFT)
        assert check.budget.strategy == "round_robin"
        assert check.budget.used_total <= check.budget.budget
    finally:
        rag.config.budget.strategy = "priority"


def test_revision_keeps_supported_corrects_contradicted_removes_unsupported(rag):
    revised = rag.check_answer(DRAFT).revised
    assert not revised.abstained
    assert "Trial Two consumed an average of 118 milliwatt-hours per day. [1]" in revised.text
    assert "Correction:" in revised.text and "1 Hz" in revised.text
    assert "Trial Two consumed more energy." in revised.text            # established part of a partial claim
    assert "direct sunlight" not in revised.text.split("(The sources do not establish")[0]
    assert revised.removed == ["The microcontroller was manufactured by Texas Instruments."]
    assert all(u.page is not None for u in revised.citations)


def test_revision_abstains_when_nothing_is_supported(rag):
    revised = rag.check_answer("The study was funded by a national research agency.").revised
    assert revised.abstained
    assert revised.text.startswith(rag.config.answer.abstain_message)


def test_answers_report_budget_use(rag):
    result = rag.ask("Why did Trial Two consume more energy than Trial One?")
    assert result.budget is not None
    assert result.budget.used_total <= result.budget.budget


def test_local_llm_answer_is_verified_and_gated(rag):
    pytest.importorskip("transformers")
    try:
        answered = rag.generate_answer("How much energy did Trial One consume per day?")
    except Exception as exc:  # pragma: no cover - model download unavailable
        pytest.skip(f"Local LLM unavailable: {exc}")
    assert answered.check is not None
    assert all(c.status == S.SUPPORTED for c in answered.check.claims)
    assert "42 milliwatt-hours" in answered.final_text and answered.check.revised.citations
    gated = rag.generate_answer("Which company manufactured the microcontroller?")
    assert gated.abstained and gated.check is None   # QA answerability gate: nothing generated
