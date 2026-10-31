"""Website mode (nothing stored), local files by path, and webpages."""

from __future__ import annotations

import http.server
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from tests.test_server import DOCS, PASSWORD, make_client, upload

LEAVE_Q = "How many paid leave days does a full-time employee receive annually?"


@pytest.fixture(scope="module")
def models():
    try:
        from carag.pipeline import ClaimAwareRAG
        ClaimAwareRAG().answerer
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Models unavailable: {exc}")


def signed_up(client, username="alice"):
    assert client.post("/api/auth/register", json={"username": username, "password": PASSWORD}).status_code == 201
    return client


def files_under(folder: Path) -> list[Path]:
    return [p for p in folder.rglob("*") if p.is_file()] if folder.exists() else []


# -- website mode ------------------------------------------------------------------------

def test_web_mode_stores_nothing_and_forgets_at_sign_out(tmp_path, models):
    client = signed_up(make_client(tmp_path, mode="web"))
    assert client.get("/api/auth/me").json()["mode"] == "web"
    assert upload(client, "01_leave_policy.txt").status_code == 201
    assert "24 days" in client.post("/api/ask", json={"question": LEAVE_Q}).json()["answer"]
    assert len(client.get("/api/history").json()["history"]) == 1

    # Nothing about the document or the question reached the disk.
    assert files_under(tmp_path / "files") == []
    db = sqlite3.connect(tmp_path / "carag.db")
    assert db.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM history").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1   # only the account

    client.post("/api/auth/logout")
    client.post("/api/auth/login", json={"username": "alice", "password": PASSWORD})
    assert client.get("/api/documents").json()["documents"] == []
    assert client.get("/api/history").json()["history"] == []
    no_docs = client.post("/api/ask", json={"question": LEAVE_Q})
    assert no_docs.status_code == 400


def test_web_mode_sessions_are_separate_even_for_one_account(tmp_path, models):
    laptop = signed_up(make_client(tmp_path, mode="web"))
    phone = make_client(tmp_path, mode="web")
    phone.post("/api/auth/login", json={"username": "alice", "password": PASSWORD})
    upload(laptop, "01_leave_policy.txt")
    assert phone.get("/api/documents").json()["documents"] == []


def test_web_mode_drops_idle_sessions(tmp_path, models):
    client = signed_up(make_client(tmp_path, mode="web", idle_timeout_s=0.5))
    upload(client, "01_leave_policy.txt")
    time.sleep(1.0)
    assert client.get("/api/documents").json()["documents"] == []


def test_web_mode_keeps_uploads_in_memory():
    from starlette.formparsers import MultiPartParser
    from carag.server.workspace import MAX_UPLOAD_BYTES
    assert MultiPartParser.spool_max_size >= MAX_UPLOAD_BYTES


def test_web_mode_refuses_local_files_and_private_addresses(tmp_path, models):
    client = signed_up(make_client(tmp_path, client_host="127.0.0.1", mode="web"))
    assert client.get("/api/auth/me").json()["local_files"] is False
    assert client.post("/api/documents/path", json={"path": str(DOCS)}).status_code == 403
    for url in ("http://localhost:8765/", "http://127.0.0.1/", "http://192.168.1.1/admin", "http://10.0.0.5/"):
        response = client.post("/api/documents/url", json={"url": url})
        assert response.status_code == 400 and "private network" in response.json()["error"], url


# -- local files ---------------------------------------------------------------------------

@pytest.fixture
def local(tmp_path, models):
    return signed_up(make_client(tmp_path / "appdata", client_host="127.0.0.1"))


def test_local_file_is_read_in_place_and_never_deleted(local, tmp_path):
    original = tmp_path / "mine" / "leave.txt"
    original.parent.mkdir()
    original.write_bytes((DOCS / "01_leave_policy.txt").read_bytes())
    assert local.get("/api/auth/me").json()["local_files"] is True

    result = local.post("/api/documents/path", json={"path": str(original)}).json()
    doc = result["added"][0]
    assert doc["origin"] == "path" and doc["location"] == str(original.resolve())
    assert files_under(tmp_path / "appdata" / "files") == []          # not copied
    assert "24 days" in local.post("/api/ask", json={"question": LEAVE_Q}).json()["answer"]

    local.delete(f"/api/documents/{doc['id']}")
    assert original.exists()                                          # only forgotten


def test_local_folder_is_added_recursively(local, tmp_path):
    folder = tmp_path / "policies"
    (folder / "travel").mkdir(parents=True)
    (folder / ".hidden").mkdir()
    (folder / "leave.txt").write_bytes((DOCS / "01_leave_policy.txt").read_bytes())
    (folder / "travel" / "travel.txt").write_bytes((DOCS / "05_travel_policy.txt").read_bytes())
    (folder / ".hidden" / "secret.txt").write_text("Hidden folder text that should be skipped.")
    (folder / "photo.jpg").write_bytes(b"\xff\xd8")
    (folder / "empty.md").write_text("")

    result = local.post("/api/documents/path", json={"path": str(folder)}).json()
    assert sorted(d["name"] for d in result["added"]) == ["leave.txt", "travel.txt"]
    assert [e["name"] for e in result["errors"]] == ["empty.md"]
    again = local.post("/api/documents/path", json={"path": str(folder)}).json()
    duplicates = {e["name"]: e["error"] for e in again["errors"] if e["name"] != "empty.md"}
    assert again["added"] == [] and set(duplicates) == {"leave.txt", "travel.txt"}
    assert all("already exists" in error for error in duplicates.values())


@pytest.mark.parametrize("path, message", [
    ("", "Enter the path"),
    ("relative/file.txt", "full path"),
    ("C:/definitely/not/here.txt", "does not exist"),
])
def test_local_path_validation(local, path, message):
    response = local.post("/api/documents/path", json={"path": path})
    assert response.status_code == 400 and message in response.json()["error"]


def test_local_files_only_from_this_computer(tmp_path, models):
    remote = signed_up(make_client(tmp_path, client_host="203.0.113.7"))
    assert remote.get("/api/auth/me").json()["local_files"] is False
    assert remote.post("/api/documents/path", json={"path": str(DOCS)}).status_code == 403


def test_local_file_survives_restart_and_reports_when_moved(tmp_path, models):
    original = tmp_path / "leave.txt"
    original.write_bytes((DOCS / "01_leave_policy.txt").read_bytes())
    first = signed_up(make_client(tmp_path / "appdata", client_host="127.0.0.1"))
    first.post("/api/documents/path", json={"path": str(original)})

    second = make_client(tmp_path / "appdata", client_host="127.0.0.1")
    second.post("/api/auth/login", json={"username": "alice", "password": PASSWORD})
    assert "24 days" in second.post("/api/ask", json={"question": LEAVE_Q}).json()["answer"]
    original.unlink()
    doc = second.get("/api/documents").json()["documents"][0]
    assert any("moved or deleted" in w for w in doc["warnings"])


# -- webpages ------------------------------------------------------------------------------

PAGE = (b"<html><head><title>Remote Work FAQ</title></head><body><article><h1>Remote work</h1>"
        b"<p>Employees may work remotely up to 3 days per week after their probation period ends.</p>"
        b"<p>Remote work requests are approved by the team lead.</p></article></body></html>")


@pytest.fixture(scope="module")
def website():
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "/secret")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(PAGE)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://localhost:{server.server_address[1]}"
    server.shutdown()


def test_local_mode_reads_a_webpage_and_keeps_it(tmp_path, models, website):
    client = signed_up(make_client(tmp_path))
    response = client.post("/api/documents/url", json={"url": f"{website}/faq"})
    assert response.status_code == 201, response.text
    doc = response.json()["document"]
    assert doc["origin"] == "url" and doc["location"] == f"{website}/faq"
    answer = client.post("/api/ask", json={"question": "How many days per week can employees work remotely?"}).json()
    assert "3 days per week" in answer["answer"] and answer["citations"][0]["url"] == f"{website}/faq"

    restarted = make_client(tmp_path)          # the page is kept even if the website changes or goes away
    restarted.post("/api/auth/login", json={"username": "alice", "password": PASSWORD})
    answer = restarted.post("/api/ask", json={"question": "Who approves remote work requests?"}).json()
    assert "team lead" in answer["answer"]
    restarted.delete(f"/api/documents/{doc['id']}")
    assert list((tmp_path / "files").rglob("*.json")) == []


def test_url_validation(tmp_path, models):
    client = signed_up(make_client(tmp_path))
    for url, message in (("", "Enter the address"), ("ftp://example.com/x", "Invalid URL"),
                         ("not a url", "Invalid URL")):
        response = client.post("/api/documents/url", json={"url": url})
        assert response.status_code == 400 and message in response.json()["error"], url


def test_every_redirect_target_is_checked(website):
    from carag.ingestion import IngestionError, load_url

    def guard(url):
        if "secret" in url:
            raise IngestionError("refused")

    with pytest.raises(IngestionError, match="refused"):
        load_url(f"{website}/redirect", url_guard=guard)
