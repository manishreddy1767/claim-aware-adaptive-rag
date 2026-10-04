from carag.claims import decompose_causal, extract_claims
from carag.query import analyze_question
from carag.text_utils import extract_numbers, split_sentences, tokenize


def test_split_sentences_respects_abbreviations_and_decimals():
    text = "The error was 0.3 degrees, e.g. in Trial One. Dr. Smith agreed. It rose to 2.8 times! Done?"
    assert split_sentences(text) == [
        "The error was 0.3 degrees, e.g. in Trial One.",
        "Dr. Smith agreed.",
        "It rose to 2.8 times!",
        "Done?",
    ]


def test_extract_numbers_normalizes():
    assert extract_numbers("It used 1,200 mAh and 3.50 V over 14 days (v2).") == {"1200", "3.5", "14"}


def test_tokenize_stems_and_drops_stopwords():
    assert tokenize("The trials consumed energy nodes node") == ["trial", "consum", "energy", "nod", "nod"]


def test_tokenize_matches_ee_plurals():
    assert tokenize("employee employees fee fees") == ["employe", "employe", "fee", "fee"]


def test_question_intents():
    assert "causal" in analyze_question("Why did Trial Two consume more energy?").intents
    assert "numeric" in analyze_question("How many spikes were recorded?").intents
    assert "comparison" in analyze_question("How does Trial One compare with Trial Two?").intents
    assert "limitation" in analyze_question("What are the limitations of the study?").intents
    assert analyze_question("Did Trial Two use 10 Hz?").is_yes_no


def test_vague_questions_are_detected():
    assert analyze_question("Why did it happen?").is_vague
    assert analyze_question("What caused it?").is_vague
    assert not analyze_question("Why did Trial Two consume more energy?").is_vague


def test_premise_extraction():
    assert analyze_question("Why did Trial One consume more energy than Trial Two?").premise == \
        "Trial One consumed more energy than Trial Two."
    assert analyze_question("Did Trial Two use a lower sampling frequency than Trial One?").premise == \
        "Trial Two used a lower sampling frequency than Trial One."
    assert analyze_question("Why was the rooftop site chosen?").premise == "The rooftop site was chosen."
    assert analyze_question("What sampling frequency did Trial One use?").premise is None


def test_premise_from_first_sentence_with_follow_up():
    analysis = analyze_question("Can employees carry forward unused leave? Explain the limits and expiration rules.")
    assert analysis.premise == "Employees can carry forward unused leave."
    assert not analysis.is_yes_no   # the follow-up asks for an explanation, not yes/no
    assert not analyze_question("Is leave paid, and how many days are given?").is_yes_no
    assert analyze_question("Can employees work remotely?").is_yes_no


def test_extract_claims_splits_only_safe_clauses():
    text = ("Trial One used 1 Hz, whereas Trial Two used 10 Hz [1]. "
            "Trial Two consumed more energy, and it also overheated. What next?")
    assert extract_claims(text) == [
        "Trial One used 1 Hz.",
        "Trial Two used 10 Hz.",
        "Trial Two consumed more energy, and it also overheated.",
    ]


def test_decompose_causal_resolves_pronoun():
    parts = decompose_causal("Trial Two consumed more energy because it used a higher sampling frequency.")
    assert parts.effect == "Trial Two consumed more energy."
    assert parts.cause == "Trial Two used a higher sampling frequency."
    noun_cause = decompose_causal("Trial Two consumed more energy due to its sampling frequency.")
    assert noun_cause.cause is None and noun_cause.effect == "Trial Two consumed more energy."
    assert decompose_causal("Trial Two used 10 Hz.") is None


def test_comparison_targets():
    analysis = analyze_question("Compare the company's annual leave policy with its work-from-home policy, "
                                "including eligibility, approval procedures, and restrictions.")
    assert analysis.comparison_targets == ["company's annual leave policy", "work-from-home policy"]
    assert analysis.comparison_aspects == "eligibility, approval procedures, and restrictions"
    assert analyze_question("What is the difference between Trial One and Trial Two?").comparison_targets == \
        ["Trial One", "Trial Two"]
    assert not analyze_question("How did the battery life of Trial One compare with Trial Two?").comparison_targets


def test_multi_part_questions():
    assert analyze_question("What is the procedure for requesting leave?").multi_part
    assert analyze_question("What benefits are available, and what are the limits?").multi_part
    assert not analyze_question("How many paid leave days does a full-time employee receive?").multi_part
