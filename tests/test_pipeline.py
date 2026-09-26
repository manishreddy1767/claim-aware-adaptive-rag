"""End-to-end question answering on the synthetic test document."""

import pytest

from carag.schema import ClaimStatus as S


def assert_citations_are_real(rag, result):
    units = {u.evidence_id: u for u in rag.index.units}
    for citation in result.citations:
        unit = units[citation.unit.evidence_id]
        assert unit.text == citation.unit.text and unit.page == citation.unit.page
    for sentence in result.sentences:
        assert sentence.citation_markers, "every answer sentence must be cited"


def test_direct_question(rag):
    result = rag.ask("What sampling frequency did Trial One use?")
    assert not result.abstained
    assert "1 Hz" in result.answer
    assert_citations_are_real(rag, result)
    assert result.citations[0].unit.page == 1
    assert all(c.status == S.SUPPORTED for c in result.claims)


def test_causal_question(rag):
    result = rag.ask("Why did Trial Two consume more energy than Trial One?")
    assert not result.abstained
    assert "higher sampling frequency" in result.answer
    assert result.premise_check.status == S.SUPPORTED
    assert_citations_are_real(rag, result)


def test_numeric_and_comparison_questions(rag):
    assert "118" in rag.ask("How much energy did Trial Two consume per day?").answer
    assert "71 days" in rag.ask("How did the battery life of Trial One compare with Trial Two?").answer


@pytest.mark.parametrize("question", [
    "What was the average wind speed at the rooftop site?",
    "Which company manufactured the microcontroller?",
    "Why did it happen?",
])
def test_abstains_without_evidence(rag, question):
    result = rag.ask(question)
    assert result.abstained
    assert result.answer == rag.config.answer.abstain_message
    assert not result.citations


def test_misleading_premise_is_corrected(rag):
    result = rag.ask("Why did Trial One consume more energy than Trial Two?")
    assert result.premise_check.status == S.CONTRADICTED
    assert "assumption" in result.answer.lower()
    assert result.citations
    assert_citations_are_real(rag, result)


def test_yes_no_question(rag):
    assert rag.ask("Did Trial Two use a lower sampling frequency than Trial One?").answer.startswith("No.")


def test_baseline_modes_run(rag):
    result = rag.ask("How was energy consumption measured?", verify=False, adaptive=False)
    assert "inline power monitor" in result.answer
    assert not result.claims


def test_empty_question_rejected(rag):
    with pytest.raises(ValueError):
        rag.ask("   ")
