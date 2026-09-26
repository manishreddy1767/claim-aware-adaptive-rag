"""Functional test of the Streamlit UI (runs the real app script headlessly)."""

from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_answers_question_end_to_end(rag, test_pdf):
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception
    assert at.title[0].value == "Claim-Aware Adaptive RAG"

    # File upload widgets are not scriptable in AppTest, so ingest via the session's pipeline.
    at.session_state.rag.add_source(str(test_pdf))
    question_box = next(t for t in at.text_input if t.label == "Question")
    question_box.input("Why did Trial Two consume more energy than Trial One?").run()
    next(b for b in at.button if b.label == "Answer").click().run()
    assert not at.exception
    answer = " ".join(i.value for i in at.info)
    assert "higher sampling frequency" in answer
    assert any("SUPPORTED" in m.value for m in at.markdown)


def test_app_reports_abstention(rag, test_pdf):
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.session_state.rag.add_source(str(test_pdf))
    next(t for t in at.text_input if t.label == "Question").input("Which company manufactured the microcontroller?").run()
    next(b for b in at.button if b.label == "Answer").click().run()
    assert not at.exception
    assert any("could not find sufficient evidence" in w.value for w in at.warning)


def test_app_verify_text_tab(rag, test_pdf):
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.session_state.rag.add_source(str(test_pdf))
    at.text_area[0].input("Trial One sampled the sensors at 10 Hz.").run()
    next(b for b in at.button if b.label == "Verify claims").click().run()
    assert not at.exception
    assert any("CONTRADICTED" in m.value for m in at.markdown)
