"""API tests for the local web application (carag/server)."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "evaluation" / "datasets" / "northstar"
PASSWORD = "correct horse"


@pytest.fixture(scope="module")
def models():
    """Load the models once for the module (skips when unavailable)."""
    try:
        from carag.pipeline import ClaimAwareRAG
        ClaimAwareRAG().answerer
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Models unavailable: {exc}")


def make_client(data_dir: Path, **settings):
    from carag.server.app import Settings, create_app
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # starlette's httpx deprecation notice
        from starlette.testclient import TestClient
    return TestClient(create_app(Settings(data_dir=data_dir, warm_up=False, **settings)))


def upload(client, name: str, data: bytes | None = None):
    data = (DOCS / name).read_bytes() if data is None else data
    return client.post("/api/documents", files={"file": (name, data, "application/octet-stream")})


@pytest.fixture
def client(tmp_path):
    return make_client(tmp_path)


@pytest.fixture
def alice(client):
    assert client.post("/api/auth/register", json={"username": "alice", "password": PASSWORD}).status_code == 201
    return client


# -- accounts ----------------------------------------------------------------------------

def test_first_run_offers_account_creation(client):
    assert client.get("/api/auth/config").json() == {"allow_signup": True, "has_users": False}
    assert client.get("/api/health").json()["status"] == "ok"


@pytest.mark.parametrize("username, password, message", [
    ("al", PASSWORD, "Usernames are 3-32"),
    ("alice smith", PASSWORD, "Usernames are 3-32"),
    ("alice", "short", "at least 8"),
    ("alice", "x" * 300, "at most 256"),
])
def test_registration_validates_input(client, username, password, message):
    response = client.post("/api/auth/register", json={"username": username, "password": password})
    assert response.status_code == 400 and message in response.json()["error"]


def test_register_signs_in_and_usernames_are_unique(alice):
    assert alice.get("/api/auth/me").json() == {"username": "alice"}
    response = alice.post("/api/auth/register", json={"username": "ALICE", "password": PASSWORD})
    assert response.status_code == 400 and "already taken" in response.json()["error"]


def test_login_logout(alice):
    assert alice.post("/api/auth/logout").status_code == 200
    assert alice.get("/api/auth/me").status_code == 401
    assert alice.get("/api/documents").status_code == 401
    bad = alice.post("/api/auth/login", json={"username": "alice", "password": "wrong password"})
    assert bad.status_code == 401 and bad.json()["error"] == "Incorrect username or password."
    unknown = alice.post("/api/auth/login", json={"username": "nobody", "password": PASSWORD})
    assert unknown.json()["error"] == "Incorrect username or password."   # same message: no user enumeration
    assert alice.post("/api/auth/login", json={"username": "Alice", "password": PASSWORD}).status_code == 200
    assert alice.get("/api/auth/me").json() == {"username": "alice"}


def test_repeated_failures_lock_the_account_temporarily(alice):
    for _ in range(5):
        alice.post("/api/auth/login", json={"username": "alice", "password": "wrong password"})
    locked = alice.post("/api/auth/login", json={"username": "alice", "password": PASSWORD})
    assert locked.status_code == 401 and "Too many failed attempts" in locked.json()["error"]


def test_passwords_are_not_stored_in_plain_text(alice, tmp_path):
    import sqlite3
    db = sqlite3.connect(tmp_path / "carag.db")
    stored = db.execute("SELECT password_hash FROM users").fetchone()[0]
    assert stored.startswith("scrypt$") and PASSWORD not in stored
    tokens = [row[0] for row in db.execute("SELECT token_hash FROM sessions")]
    assert alice.cookies.get("carag_session") not in tokens   # only hashes are stored


def test_session_cookie_is_http_only(client):
    response = client.post("/api/auth/register", json={"username": "alice", "password": PASSWORD})
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie


def test_signup_can_be_closed_after_the_first_account(tmp_path):
    client = make_client(tmp_path, allow_signup=False)
    assert client.get("/api/auth/config").json()["allow_signup"] is True        # first account
    assert client.post("/api/auth/register", json={"username": "admin", "password": PASSWORD}).status_code == 201
    assert client.get("/api/auth/config").json()["allow_signup"] is False
    second = client.post("/api/auth/register", json={"username": "bob", "password": PASSWORD})
    assert second.status_code == 403


def test_malformed_json_is_rejected(client):
    response = client.post("/api/auth/login", content=b"not json", headers={"Content-Type": "application/json"})
    assert response.status_code == 400
    assert client.post("/api/auth/login", json=["a", "list"]).status_code == 400


# -- security ----------------------------------------------------------------------------

def test_cross_site_requests_are_rejected(alice):
    response = alice.post("/api/ask", json={"question": "x"}, headers={"Origin": "http://evil.example"})
    assert response.status_code == 403


def test_security_headers(client):
    headers = client.get("/").headers
    assert "default-src 'self'" in headers["content-security-policy"]
    assert headers["x-frame-options"] == "DENY" and headers["x-content-type-options"] == "nosniff"


def test_unknown_api_route_is_json_404(alice):
    response = alice.get("/api/nope")
    assert response.status_code == 404 and response.json()["error"] == "Not found."


# -- documents ---------------------------------------------------------------------------

def test_upload_validation(alice, models):
    assert upload(alice, "notes.exe", b"MZ...").status_code == 400
    empty = upload(alice, "empty.txt", b"")
    assert empty.status_code == 400 and "empty" in empty.json()["error"]
    assert alice.post("/api/documents", data={"x": "1"}).status_code == 400


def test_upload_list_duplicate_delete(alice, models):
    response = upload(alice, "01_leave_policy.txt")
    assert response.status_code == 201
    doc = response.json()["document"]
    assert doc["name"] == "01_leave_policy.txt" and doc["sentences"] > 5
    duplicate = upload(alice, "01_leave_policy.txt")
    assert duplicate.status_code == 400 and "already exists" in duplicate.json()["error"]
    assert [d["id"] for d in alice.get("/api/documents").json()["documents"]] == [doc["id"]]
    assert alice.delete(f"/api/documents/{doc['id']}").status_code == 200
    assert alice.get("/api/documents").json()["documents"] == []
    assert alice.delete(f"/api/documents/{doc['id']}").status_code == 404


def test_file_names_cannot_escape_the_data_folder(alice, models, tmp_path):
    response = upload(alice, "../../outside.txt", (DOCS / "01_leave_policy.txt").read_bytes())
    assert response.status_code == 201 and response.json()["document"]["name"] == "outside.txt"
    assert not (tmp_path.parent / "outside.txt").exists()
    stored = list((tmp_path / "files").rglob("*.txt"))
    assert len(stored) == 1 and stored[0].is_relative_to(tmp_path / "files")


def test_text_without_sentences_is_rejected(alice, models):
    response = upload(alice, "blank.txt", b"   \n\n  ")
    assert response.status_code == 400


# -- questions ---------------------------------------------------------------------------

def test_ask_requires_documents_and_a_question(alice, models):
    no_docs = alice.post("/api/ask", json={"question": "How many leave days?"})
    assert no_docs.status_code == 400 and "Upload at least one document" in no_docs.json()["error"]
    upload(alice, "01_leave_policy.txt")
    assert alice.post("/api/ask", json={"question": "   "}).status_code == 400
    assert alice.post("/api/ask", json={"question": "x" * 1001}).status_code == 400


def test_ask_returns_a_cited_answer_and_records_history(alice, models):
    upload(alice, "01_leave_policy.txt")
    view = alice.post("/api/ask", json={"question": "How many paid leave days does a full-time employee "
                                                    "receive annually?"}).json()
    assert view["outcome"] == "answer" and "24 days" in view["answer"] and "[1]" in view["answer"]
    assert view["citations"][0]["source"] == "01_leave_policy.txt"
    assert view["claims"] and view["claims"][0]["status"] == "SUPPORTED"
    history = alice.get("/api/history").json()["history"]
    assert history[0]["question"].startswith("How many paid leave days") and history[0]["outcome"] == "answer"
    assert alice.delete("/api/history").status_code == 200
    assert alice.get("/api/history").json()["history"] == []


@pytest.mark.parametrize("question, outcome", [
    ("Does the company guarantee 30 days of paid annual leave to every employee?", "reject_premise"),
    ("What is the company's maternity leave duration in weeks?", "not_specified"),
    ("What is the company's stock option vesting schedule?", "abstain"),
])
def test_outcomes_reach_the_ui(alice, models, question, outcome):
    for name in ("01_leave_policy.txt", "04_attendance_and_parental_leave.txt"):
        upload(alice, name)
    assert alice.post("/api/ask", json={"question": question}).json()["outcome"] == outcome


def test_verify_text(alice, models):
    upload(alice, "01_leave_policy.txt")
    result = alice.post("/api/verify", json={"text": "Full-time employees receive 30 days of paid leave."}).json()
    assert result["claims"][0]["status"] == "CONTRADICTED"
    assert alice.post("/api/verify", json={"text": ""}).status_code == 400


def test_users_cannot_see_each_others_documents(tmp_path, models):
    alice, bob = make_client(tmp_path), make_client(tmp_path)
    alice.post("/api/auth/register", json={"username": "alice", "password": PASSWORD})
    bob.post("/api/auth/register", json={"username": "bob", "password": PASSWORD})
    doc = upload(alice, "01_leave_policy.txt").json()["document"]
    assert bob.get("/api/documents").json()["documents"] == []
    assert bob.delete(f"/api/documents/{doc['id']}").status_code == 404
    assert bob.post("/api/ask", json={"question": "How many leave days?"}).status_code == 400
    upload(bob, "07_it_equipment_policy.txt")
    answer = bob.post("/api/ask", json={"question": "How many paid leave days does a full-time employee "
                                                  "receive annually?"}).json()
    assert "24 days" not in answer["answer"]
    assert alice.get("/api/history").json()["history"] == []


def test_documents_and_accounts_survive_a_restart(tmp_path, models):
    first = make_client(tmp_path)
    first.post("/api/auth/register", json={"username": "alice", "password": PASSWORD})
    upload(first, "01_leave_policy.txt")
    second = make_client(tmp_path)   # a new server process on the same data folder
    assert second.post("/api/auth/login", json={"username": "alice", "password": PASSWORD}).status_code == 200
    assert [d["name"] for d in second.get("/api/documents").json()["documents"]] == ["01_leave_policy.txt"]
    view = second.post("/api/ask", json={"question": "When must carried-forward leave days be used?"}).json()
    assert "March 31" in view["answer"]
