"""Local AI models: any server on the computer is found and used; without one, RAG answers.

Fake servers (an Ollama-style one and an OpenAI-compatible one like LM Studio) return fixed
replies, so these tests check how answers are handled, not what a real model would write.
"""

from __future__ import annotations

import http.server
import json
import socket
import threading

import pytest

from tests.test_server import PASSWORD, make_client, upload

LEAVE_Q = "How many paid leave days does a full-time employee receive annually?"
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


def fake_server(kind: str, model_names: list[str], fail: bool = False):
    """kind: "ollama" (/api/tags, /api/chat) or "openai" (/v1/models, /v1/chat/completions)."""
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
            if kind == "ollama" and self.path == "/api/tags":
                self._send({"models": [{"name": n} for n in model_names]})
            elif kind == "openai" and self.path == "/v1/models":
                self._send({"object": "list", "data": [{"id": n, "object": "model"} for n in model_names]})
            else:
                self._send({"error": "not found"}, 404)

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(request)
            if fail:
                self._send({"error": "out of memory"}, 500)
                return
            question = request["messages"][-1]["content"]
            reply = "<think>hidden</think>" + next((r for k, r in REPLIES.items() if k in question), "Not sure.")
            if kind == "ollama":
                self._send({"message": {"role": "assistant", "content": reply}})
            else:
                self._send({"choices": [{"message": {"role": "assistant", "content": reply}}]})

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    server.seen = seen
    server.url = f"http://127.0.0.1:{server.server_address[1]}"
    return server


@pytest.fixture(scope="module")
def ollama():
    server = fake_server("ollama", ["fake:1b", "nomic-embed-text:latest"])
    yield server
    server.shutdown()


@pytest.fixture(scope="module")
def lmstudio():
    server = fake_server("openai", ["qwen3-8b-instruct"])
    yield server
    server.shutdown()


def closed_port() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{sock.getsockname()[1]}"


def client_with(tmp_path, servers, mode="local"):
    """An app whose AI discovery only looks at the given (name, url, kind) servers."""
    client = make_client(tmp_path, client_host="127.0.0.1", mode=mode)
    client.app.state.workspaces.llm_candidates = servers
    client.post("/api/auth/register", json={"username": "alice", "password": PASSWORD})
    return client


# -- no AI on the computer: RAG answers ------------------------------------------------------

def test_without_any_ai_the_documents_answer(tmp_path, models):
    client = client_with(tmp_path, [("Ollama", closed_port(), "ollama"), ("LM Studio", closed_port(), "openai")])
    status = client.get("/api/ai").json()
    assert status["enabled"] is True and status["active"] is None and status["servers"] == []
    assert "No local AI model was found" in status["message"]
    upload(client, "01_leave_policy.txt")
    view = client.post("/api/ask", json={"question": LEAVE_Q}).json()
    assert view["outcome"] == "answer" and "24 days" in view["answer"] and "ai" not in view
    asked_for_ai = client.post("/api/ask", json={"question": LEAVE_Q, "style": "ai"}).json()
    assert "24 days" in asked_for_ai["answer"] and "No local AI model is available" in asked_for_ai["fallback"]


# -- any AI found is used ----------------------------------------------------------------------

def test_ollama_is_found_and_used_automatically(tmp_path, models, ollama):
    client = client_with(tmp_path, [("Ollama", ollama.url, "ollama")])
    status = client.get("/api/ai").json()
    assert status["servers"][0]["models"] == ["fake:1b"]                 # embedding model left out
    assert status["active"] == {"server": "Ollama", "url": ollama.url, "model": "fake:1b"}
    upload(client, "01_leave_policy.txt")
    view = client.post("/api/ask", json={"question": LEAVE_Q}).json()   # no style: automatic
    assert view["ai"]["model"] == "fake:1b" and view["ai"]["provider"] == "Ollama"
    assert "24 days" in view["answer"] and "sick leave" not in view["answer"]
    assert any("sick leave" in c["claim"] for c in view["removed_claims"])
    assert "<think>" not in view["ai"]["draft"]
    sent = ollama.seen[-1]
    assert sent["think"] is False and sent["options"]["temperature"] == 0
    assert "Full-time employees receive 24 days" in sent["messages"][-1]["content"]


def test_openai_compatible_servers_are_used(tmp_path, models, lmstudio):
    client = client_with(tmp_path, [("Ollama", closed_port(), "ollama"), ("LM Studio", lmstudio.url, "openai")])
    assert client.get("/api/ai").json()["active"]["server"] == "LM Studio"
    upload(client, "01_leave_policy.txt")
    view = client.post("/api/ask", json={"question": LEAVE_Q, "style": "ai"}).json()
    assert view["ai"] == {"model": "qwen3-8b-instruct", "provider": "LM Studio", "draft": view["ai"]["draft"]}
    assert "24 days" in view["answer"] and "sick leave" not in view["answer"]
    assert lmstudio.seen[-1]["temperature"] == 0 and lmstudio.seen[-1]["model"] == "qwen3-8b-instruct"


def test_the_best_known_model_is_chosen_across_servers(tmp_path, models, ollama, lmstudio):
    client = client_with(tmp_path, [("Ollama", ollama.url, "ollama"), ("LM Studio", lmstudio.url, "openai")])
    # qwen3 is preferred over an unknown model, even on the second server.
    assert client.get("/api/ai").json()["active"]["model"] == "qwen3-8b-instruct"
    chosen = client.put("/api/ai", json={"enabled": True, "model": "fake:1b"}).json()
    assert chosen["active"] == {"server": "Ollama", "url": ollama.url, "model": "fake:1b"}
    missing = client.put("/api/ai", json={"enabled": True, "model": "gone:70b"}).json()
    assert missing["active"]["model"] == "qwen3-8b-instruct" and "'gone:70b' was not found" in missing["message"]


def test_a_failing_model_falls_back_to_the_documents(tmp_path, models):
    broken = fake_server("ollama", ["fake:1b"], fail=True)
    try:
        client = client_with(tmp_path, [("Ollama", broken.url, "ollama")])
        upload(client, "01_leave_policy.txt")
        response = client.post("/api/ask", json={"question": LEAVE_Q})
        assert response.status_code == 200
        view = response.json()
        assert "24 days" in view["answer"] and "ai" not in view
        assert "could not answer" in view["fallback"] and "out of memory" in view["fallback"]
    finally:
        broken.shutdown()


def test_ai_can_be_turned_off_or_skipped(tmp_path, models, ollama):
    client = client_with(tmp_path, [("Ollama", ollama.url, "ollama")])
    upload(client, "01_leave_policy.txt")
    assert "ai" not in client.post("/api/ask", json={"question": LEAVE_Q, "style": "quotes"}).json()
    off = client.put("/api/ai", json={"enabled": False, "model": "auto"}).json()
    assert off["active"] is None and "off" in off["message"]
    assert "ai" not in client.post("/api/ask", json={"question": LEAVE_Q}).json()
    saved = json.loads((tmp_path / "settings.json").read_text())["ai"]
    assert saved == {"enabled": False, "model": "auto", "server": None}


# -- how AI answers are checked ------------------------------------------------------------------

def test_ai_answer_shows_both_sides_of_a_conflict(tmp_path, models, ollama):
    client = client_with(tmp_path, [("Ollama", ollama.url, "ollama")])
    upload(client, "05_travel_policy.txt")
    upload(client, "06_travel_faq_2026.txt")
    view = client.post("/api/ask", json={"question": "What is the daily meal allowance for domestic travel?"}).json()
    assert view["outcome"] == "conflict" and view["answer"].count("The sources disagree") == 1
    assert "2,500" in view["answer"] and "3,000" in view["answer"]


def test_not_in_documents_and_trailing_disclaimers(tmp_path, models, ollama):
    client = client_with(tmp_path, [("Ollama", ollama.url, "ollama")])
    for name in ("01_leave_policy.txt", "07_it_equipment_policy.txt"):
        upload(client, name)
    assert client.post("/api/ask", json={"question": "What is the stock option vesting schedule?"}).json()[
        "outcome"] == "abstain"
    laptop = client.post("/api/ask", json={"question": "Does the company give every employee a new laptop "
                                                       "every year?"}).json()
    assert laptop["outcome"] == "reject_premise" and "every 3 years" in laptop["answer"]
    yes_prefix = client.post("/api/ask", json={"question": "What approval does an employee need for a "
                                                           "12-day leave?"}).json()
    assert yes_prefix["outcome"] == "answer" and "additional approval from HR" in yes_prefix["answer"]


def test_named_subject_missing_is_refused_before_calling_the_model(tmp_path, models, ollama):
    client = client_with(tmp_path, [("Ollama", ollama.url, "ollama")])
    upload(client, "01_leave_policy.txt")
    calls = len(ollama.seen)
    view = client.post("/api/ask", json={"question": "Who is Elizabeth Bennet?"}).json()
    assert view["outcome"] == "abstain" and "Elizabeth Bennet" in view["answer"]
    assert len(ollama.seen) == calls


# -- settings ---------------------------------------------------------------------------------------

def test_website_visitors_cannot_change_ai_settings(tmp_path, models, ollama):
    client = client_with(tmp_path, [("Ollama", ollama.url, "ollama")], mode="web")
    assert client.get("/api/ai").json()["editable"] is False
    assert client.put("/api/ai", json={"enabled": False}).status_code == 403


def test_custom_server_address_and_validation(tmp_path, models):
    from carag.llm import known_servers
    assert known_servers("127.0.0.1:9999/v1")[0] == ("Custom server", "http://127.0.0.1:9999", "openai")
    client = client_with(tmp_path, [])
    bad = client.put("/api/ai", json={"enabled": True, "model": "auto", "server": "http://"})
    assert bad.status_code == 400
    ok = client.put("/api/ai", json={"enabled": True, "model": "auto", "server": "http://127.0.0.1:9999"})
    assert ok.status_code == 200 and ok.json()["server"] == "http://127.0.0.1:9999"
