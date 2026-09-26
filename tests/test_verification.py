import pytest

from carag.schema import ClaimStatus as S


@pytest.mark.parametrize("claim, expected", [
    ("Trial Two sampled the sensors at 10 Hz.", S.SUPPORTED),
    ("Both trials used the same firmware version.", S.SUPPORTED),
    ("The reboots of Node B were caused by a loose power connector.", S.SUPPORTED),
    ("Trial One sampled the sensors at 10 Hz.", S.CONTRADICTED),
    ("The reboots were caused by the sampling configuration.", S.CONTRADICTED),
    ("The battery in Trial Two was projected to last 71 days.", S.CONTRADICTED),
    ("The microcontroller was manufactured by Texas Instruments.", S.INSUFFICIENT_EVIDENCE),
    ("The study was funded by a national research agency.", S.INSUFFICIENT_EVIDENCE),
    ("Trial Two consumed more energy because its nodes were placed in direct sunlight.",
     S.PARTIALLY_SUPPORTED),
])
def test_claim_labels(rag, claim, expected):
    result = rag.verify_claim(claim)
    assert result.status == expected, result.explanation


def test_supported_claim_records_citable_evidence(rag):
    result = rag.verify_claim("Trial One consumed an average of 42 milliwatt-hours per day.")
    assert result.status == S.SUPPORTED
    top = result.supporting[0]
    assert top.evidence_ids[0].startswith("claim_aware_rag_test_document.pdf#p2.")
    assert "42 milliwatt-hours" in top.premise


def test_missing_evidence_is_not_labelled_contradicted(rag):
    result = rag.verify_claim("The nodes were enclosed in waterproof plastic housings.")
    assert result.status != S.CONTRADICTED


def test_verify_text_splits_and_checks_each_claim(rag):
    results = rag.verify_text("Trial One used 1 Hz sampling, whereas Trial Two used 50 Hz sampling.")
    assert len(results) == 2
    assert results[0].status == S.SUPPORTED
    assert results[1].status in (S.CONTRADICTED, S.UNCERTAIN)
