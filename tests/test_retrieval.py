from carag.ingestion import load_bytes

from .conftest import unit_containing


def top_texts(result, n=3):
    return [e.unit.text for e in result.evidence[:n]]


def test_direct_question_retrieves_gold_sentence_first(rag):
    result = rag.retrieve("What sampling frequency did Trial One use?")
    assert result.sufficient
    assert "Trial One sampled the sensors at a frequency of 1 Hz." == result.evidence[0].unit.text


def test_causal_question_prefers_relevant_cause_over_keyword_distractors(rag):
    result = rag.retrieve("Why did Trial Two consume more energy than Trial One?")
    assert result.sufficient
    gold = unit_containing(rag, "caused by its higher sampling frequency").evidence_id
    assert gold in [e.unit.evidence_id for e in result.evidence[:3]]
    # Sentences with "because" but about other topics must not get an intent bonus.
    site = next((e for e in result.evidence if "line of sight" in e.unit.text), None)
    assert site is None or site.bonus == 0.0


def test_numeric_question(rag):
    result = rag.retrieve("How many short pollution spikes did Trial Two record?")
    assert "12 short pollution spikes" in result.evidence[0].unit.text


def test_absent_information_is_insufficient(rag):
    result = rag.retrieve("What was the average wind speed at the rooftop site?")
    assert not result.sufficient
    assert result.expanded           # adaptive retrieval tried to expand first
    assert len(result.trace) >= 2


def test_adaptive_retrieval_trims_weak_evidence(rag):
    result = rag.retrieve("How was energy consumption measured?")
    assert result.refined and len(result.evidence) < rag.config.retrieval.max_k
    assert "inline power monitor" in result.evidence[0].unit.text


def test_near_duplicate_evidence_is_suppressed(rag):
    from carag.pipeline import ClaimAwareRAG
    system = ClaimAwareRAG()
    system.add_document(load_bytes(
        b"Trial Two consumed 118 milliwatt-hours per day.\n\n"
        b"Trial Two consumed 118 milliwatt hours each day.\n\n"
        b"The gateway was on the roof.\n", "dup.txt"))
    result = system.retrieve("How much energy did Trial Two consume per day?")
    texts = [e.unit.text for e in result.evidence]
    assert sum("118" in t for t in texts) == 1


def test_evidence_keeps_source_metadata(rag):
    result = rag.retrieve("How much energy did Trial Two consume per day?")
    unit = result.evidence[0].unit
    assert unit.source == "claim_aware_rag_test_document.pdf" and unit.page == 2
