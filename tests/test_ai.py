"""Local LLM answers (Ollama): drafting, verification of the draft, settings and errors.

A fake Ollama server returns fixed replies, so these tests check how answers are handled,
not what a real model would write.
"""

from __future__ import annotations

import http.server
import json
import threading

import pytest

from tests.test_server import DOCS, PASSWORD, make_client, upload

REPLIES = {
    # One supported statement and one invented one, which verification must remove.
    "How many paid leave days": "Full-time employees receive 24 days of paid leave per calendar year. "
                                "Employees also receive unlimited paid sick leave every year.",
    "meal allowance": "The daily meal allowance for domestic travel is ₹2,500. "
                      "The daily meal allowance for domestic travel is ₹3,000.",
    "stock option": "The documents do not contain this information.",
    "laptop every year": "No. Laptops are replaced every 3 years. The documents do not contain this information.",
    "12-day leave": "Yes. Leave of more than 10 consecutive days requires additional approval from HR.",
}


@pytest.fixture(scope="module")
def models():
    try:
        from carag.pipeline import ClaimAwareRAG
        ClaimAwareRAG().answerer
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Models unavailable: {exc}")


@pytest.fixture(scope="module")
def fake_ollama():
    seen = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, payload, status=200):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send({"models": [{"name": "fake:1b"}, {"name": "other:2b"}]})

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(request)
            if request["model"] not in ("fake:1b", "other:2b"):
                self._send({"error": "model not found"}, 404)
                return
            question = request["messages"][-1]["content"]
            reply = next((r for key, r in REPLIES.items() if key in question), "I am not sure.")
            self._send({"message": {"role": "assistant", "content": f"<think>hidden</think>{reply}"}})

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    server.seen = seen
    server.url = f"http://127.0.0.1:{server.server_address[1]}"
    yield server
    server.shutdown()


def ai_client(tmp_path, fake_ollama, mode="local", enable=True):
    from carag.llm import OllamaClient
    client = make_client(tmp_path, client_host="127.0.0.1", mode=mode)
    client.app.state.workspaces.ollama = OllamaClient(fake_ollama.url)
    client.post("/api/auth/register", json={"username": "alice", "password": PASSWORD})
    if enable:
        assert client.put("/api/ai", json={"enabled": True, "model": "fake:1b"}).status_code == 200
    return client


def test_ai_is_off_by_default_and_reports_ollama_status(tmp_path, models, fake_ollama):
    client = ai_client(tmp_path, fake_ollama, enable=False)
    status = client.get("/api/ai").json()
    assert status["enabled"] is False and status["model"] == "qwen3:8b"
    assert status["running"] is True and status["model_ready"] is False
    assert status["installed"] == ["fake:1b", "other:2b"] and "ollama pull qwen3:8b" in status["message"]
    upload(client, "01_leave_policy.txt")
    off = client.post("/api/ask", json={"question": "How many paid leave days?", "style": "ai"})
    assert off.status_code == 400 and "turned off" in off.json()["error"]


def test_settings_are_saved(tmp_path, models, fake_ollama):
    client = ai_client(tmp_path, fake_ollama)
    status = client.get("/api/ai").json()
    assert status["enabled"] is True and status["model"] == "fake:1b" and status["model_ready"] is True
    assert json.loads((tmp_path / "settings.json").read_text())["ai"] == {"enabled": True, "model": "fake:1b"}


def test_invented_statements_are_removed_from_ai_answers(tmp_path, models, fake_ollama):
    client = ai_client(tmp_path, fake_ollama)
    upload(client, "01_leave_policy.txt")
    view = client.post("/api/ask", json={"question": "How many paid leave days does a full-time employee "
                                                     "receive annually?", "style": "ai"}).json()
    assert view["outcome"] == "answer" and view["ai"]["model"] == "fake:1b"
    assert "24 days" in view["answer"] and "[1]" in view["answer"]
    assert "sick leave" not in view["answer"]                       # removed by verification
    assert any("sick leave" in c["claim"] for c in view["removed_claims"])
    assert "sick leave" in view["ai"]["draft"] and "<think>" not in view["ai"]["draft"]
    assert view["citations"][0]["source"] == "01_leave_policy.txt"
    sent = fake_ollama.seen[-1]
    assert sent["think"] is False and sent["options"]["temperature"] == 0
    assert "Full-time employees receive 24 days" in sent["messages"][-1]["content"]   # evidence was given
    assert client.get("/api/history").json()["history"][0]["result"]["ai"]["model"] == "fake:1b"


def test_ai_answer_shows_both_sides_of_a_conflict(tmp_path, models, fake_ollama):
    client = ai_client(tmp_path, fake_ollama)
    upload(client, "05_travel_policy.txt")
    upload(client, "06_travel_faq_2026.txt")
    view = client.post("/api/ask", json={"question": "What is the daily meal allowance for domestic travel?",
                                         "style": "ai"}).json()
    assert view["outcome"] == "conflict"
    assert view["answer"].count("The sources disagree") == 1
    assert "2,500" in view["answer"] and "3,000" in view["answer"]


def test_not_in_documents_and_trailing_disclaimers(tmp_path, models, fake_ollama):
    client = ai_client(tmp_path, fake_ollama)
    for name in ("01_leave_policy.txt", "07_it_equipment_policy.txt"):
        upload(client, name)
    absent = client.post("/api/ask", json={"question": "What is the stock option vesting schedule?",
                                           "style": "ai"}).json()
    assert absent["outcome"] == "abstain"
    laptop = client.post("/api/ask", json={"question": "Does the company give every employee a new laptop "
                                                       "every year?", "style": "ai"}).json()
    assert laptop["outcome"] == "reject_premise" and "every 3 years" in laptop["answer"]
    yes_prefix = client.post("/api/ask", json={"question": "What approval does an employee need for a "
                                                           "12-day leave?", "style": "ai"}).json()
    assert yes_prefix["outcome"] == "answer"   # "Yes." on a non yes/no question is not a confirmation
    assert "additional approval from HR" in yes_prefix["answer"]


def test_named_subject_missing_is_refused_before_calling_the_model(tmp_path, models, fake_ollama):
    client = ai_client(tmp_path, fake_ollama)
    upload(client, "01_leave_policy.txt")
    calls = len(fake_ollama.seen)
    view = client.post("/api/ask", json={"question": "Who is Elizabeth Bennet?", "style": "ai"}).json()
    assert view["outcome"] == "abstain" and "Elizabeth Bennet" in view["answer"]
    assert len(fake_ollama.seen) == calls


def test_ollama_problems_are_reported(tmp_path, models, fake_ollama):
    from carag.llm import OllamaClient
    client = ai_client(tmp_path, fake_ollama)
    upload(client, "01_leave_policy.txt")
    client.put("/api/ai", json={"enabled": True, "model": "missing:7b"})
    missing = client.post("/api/ask", json={"question": "How many paid leave days?", "style": "ai"})
    assert missing.status_code == 400 and "ollama pull missing:7b" in missing.json()["error"]

    client.app.state.workspaces.ollama = OllamaClient("http://127.0.0.1:9")   # nothing listens there
    status = client.get("/api/ai").json()
    assert status["running"] is False and "not running" in status["message"]
    down = client.post("/api/ask", json={"question": "How many paid leave days?", "style": "ai"})
    assert down.status_code == 400 and "Could not reach Ollama" in down.json()["error"]


def test_website_visitors_cannot_change_ai_settings(tmp_path, models, fake_ollama):
    client = ai_client(tmp_path, fake_ollama, mode="web", enable=False)
    assert client.get("/api/ai").json()["editable"] is False
    assert client.put("/api/ai", json={"enabled": True, "model": "fake:1b"}).status_code == 403
