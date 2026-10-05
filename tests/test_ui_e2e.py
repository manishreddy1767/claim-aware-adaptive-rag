"""Browser end-to-end tests: a real server and a real (headless) Chromium.

Skipped when Playwright or its Chromium build is not installed
(pip install playwright && python -m playwright install chromium).
"""

from __future__ import annotations

import re
import socket
import threading
import time
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "evaluation" / "datasets" / "northstar"
PASSWORD = "correct horse"

playwright_api = pytest.importorskip("playwright.sync_api")
expect = playwright_api.expect


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _serve(data_dir: Path, mode: str):
    """Start the application on a free port; returns (url, stop)."""
    try:
        from carag.pipeline import ClaimAwareRAG
        ClaimAwareRAG().answerer
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Models unavailable: {exc}")
    import uvicorn
    from carag.server.app import Settings, create_app

    port = _free_port()
    app = create_app(Settings(data_dir=data_dir, mode=mode))
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.1)

    def stop():
        srv.should_exit = True
        thread.join(timeout=10)
    return f"http://127.0.0.1:{port}/", stop


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """The local application (documents saved) with an empty data folder."""
    url, stop = _serve(tmp_path_factory.mktemp("appdata"), "local")
    yield url
    stop()


@pytest.fixture(scope="module")
def web_server(tmp_path_factory):
    """The hosted-website mode: documents kept in memory for one sign-in only."""
    url, stop = _serve(tmp_path_factory.mktemp("webdata"), "web")
    yield url
    stop()


@pytest.fixture(scope="module")
def webpage():
    """A small website on this computer to add as a source."""
    import http.server

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"<html><head><title>Parking</title></head><body><article><h1>Parking</h1>"
                             b"<p>Employees can park in the basement garage free of charge.</p>"
                             b"<p>Visitors must register at reception before parking.</p></article></body></html>")

        def log_message(self, *args):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://localhost:{srv.server_address[1]}/parking"
    srv.shutdown()


@pytest.fixture(scope="module")
def browser():
    with playwright_api.sync_playwright() as pw:
        try:
            chromium = pw.chromium.launch()
        except Exception as exc:  # pragma: no cover - environment dependent
            pytest.skip(f"Chromium unavailable: {exc}")
        yield chromium
        chromium.close()


@pytest.fixture
def page(browser):
    context = browser.new_context()
    page = context.new_page()
    errors: list[str] = []
    page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    page.on("dialog", lambda dialog: dialog.accept())   # confirm() prompts
    page.errors = errors
    yield page
    context.close()
    # The browser logs every 4xx response (wrong password, rejected upload...) as a failed load;
    # those are expected. JavaScript errors and CSP violations ("Refused to ...") are not.
    unexpected = [e for e in errors if not e.startswith("Failed to load resource: the server responded "
                                                        "with a status of 4")]
    assert not unexpected, f"Browser errors: {unexpected}"


def register(page, url: str, username: str) -> None:
    page.goto(url)
    page.get_by_role("tab", name="Create account").click()
    page.get_by_label("Username").fill(username)
    page.get_by_label("Password").fill(PASSWORD)
    page.locator("#auth-submit").click()
    expect(page.locator("#user-name")).to_have_text(username)


def upload(page, *names: str) -> None:
    page.locator("#file-input").set_input_files([str(DOCS / n) for n in names])
    for name in names:
        expect(page.locator("#doc-list")).to_contain_text(name, timeout=60_000)


def ask(page, question: str):
    page.get_by_label("Your question").fill(question)
    page.get_by_role("button", name="Ask", exact=True).click()
    card = page.locator("#answers article.card").first
    expect(card.locator(".card-question")).to_have_text(question, timeout=120_000)
    return card


def test_first_run_signup_validation_and_main_flow(page, server):
    page.goto(server)
    expect(page.get_by_role("heading", name="Create your account")).to_be_visible()
    page.get_by_label("Username").fill("alice")
    page.get_by_label("Password").fill("short")
    page.locator("#auth-submit").click()
    expect(page.locator("#auth-error")).to_contain_text("at least 8")

    page.get_by_label("Password").fill(PASSWORD)
    page.locator("#auth-submit").click()
    expect(page.locator("#user-name")).to_have_text("alice")
    expect(page.locator("#model-status")).to_contain_text("Ready", timeout=180_000)
    expect(page.locator("#doc-empty")).to_be_visible()

    upload(page, "01_leave_policy.txt", "02_work_from_home_policy.txt")
    expect(page.locator("#doc-count")).to_have_text("2")

    card = ask(page, "How many paid leave days does a full-time employee receive annually?")
    expect(card.locator(".pill")).to_have_text("Answered from your documents")
    expect(card.locator(".answer-text")).to_contain_text("24 days")
    expect(card.locator(".source-where").first).to_contain_text("01_leave_policy.txt")
    card.locator(".cite").first.click()
    expect(card.locator(".source").first).to_have_class(re.compile("flash"))
    card.get_by_text("How this answer was checked").click()
    expect(card.locator(".claim-status").first).to_have_text("Supported")

    card = ask(page, "Does the company guarantee 30 days of paid annual leave to every employee?")
    expect(card.locator(".pill")).to_have_text("The question's assumption is wrong")

    card = ask(page, "What is the company's stock option vesting schedule?")
    expect(card.locator(".pill")).to_have_text("No answer in your documents")


def _register_via_api(server: str, username: str) -> None:
    import json
    import urllib.error
    import urllib.request
    request = urllib.request.Request(server + "api/auth/register", method="POST",
                                     data=json.dumps({"username": username, "password": PASSWORD}).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(request, timeout=10)
    except urllib.error.HTTPError as exc:   # already registered by an earlier run of this test
        assert "already taken" in exc.read().decode()


def test_tab_choice_survives_a_slow_config_reply(page, server):
    """Clicking 'Create account' before /api/auth/config answers must not be undone by the answer."""
    _register_via_api(server, "existing_user")   # with accounts present the reply selects 'Sign in'
    held = []
    page.route("**/api/auth/config", lambda route: held.append(route))
    page.goto(server)
    page.wait_for_function("document.querySelector('#auth-view') && !document.querySelector('#auth-view').hidden")
    page.get_by_role("tab", name="Create account").click()
    for _ in range(50):
        if held:
            break
        page.wait_for_timeout(100)
    assert held, "the config request was not intercepted"
    for route in held:
        route.continue_()
    page.unroute("**/api/auth/config")
    page.wait_for_timeout(300)
    expect(page.get_by_role("heading", name="Create your account")).to_be_visible()
    page.get_by_label("Username").fill("slowpoke")
    page.get_by_label("Password").fill(PASSWORD)
    page.locator("#auth-submit").click()
    expect(page.locator("#user-name")).to_have_text("slowpoke")


def test_upload_errors_are_shown(page, server):
    register(page, server, "uploader")
    page.locator("#file-input").set_input_files(
        [{"name": "program.exe", "mimeType": "application/octet-stream", "buffer": b"MZ"}])
    expect(page.locator("#uploads li.error")).to_contain_text("not a supported file type")
    page.locator("#file-input").set_input_files(
        [{"name": "empty.txt", "mimeType": "text/plain", "buffer": b""}])
    expect(page.locator("#uploads li.error").last).to_contain_text("empty")
    page.get_by_label("Your question").fill("Anything?")
    page.get_by_role("button", name="Ask", exact=True).click()
    expect(page.locator("#ask-error")).to_contain_text("Add at least one document")


def test_document_content_cannot_inject_html(page, server):
    register(page, server, "mallory")
    payload = (b"Security Notice\n\nThe <img src=x onerror=\"document.title='pwned'\"> badge policy requires "
               b"badges at all times. Visitors must wear a <script>document.title='pwned'</script> visitor badge.")
    page.locator("#file-input").set_input_files(
        [{"name": "<b onmouseover=alert(1)>notes.txt", "mimeType": "text/plain", "buffer": payload}])
    expect(page.locator("#doc-list")).to_contain_text("<b onmouseover=alert(1)>notes.txt", timeout=60_000)
    card = ask(page, "What does the badge policy require?")
    expect(card).to_contain_text("<img src=x")      # shown as text
    assert card.locator("img").count() == 0 and page.locator("#doc-list b").count() == 0
    assert page.title() == "Claim-Aware RAG"


def test_history_check_text_delete_logout_login(page, server):
    register(page, server, "bob")
    upload(page, "01_leave_policy.txt")
    ask(page, "When must carried-forward leave days be used?")

    page.get_by_role("tab", name="History").click()
    item = page.locator(".history-item").first
    expect(item).to_contain_text("When must carried-forward leave days be used?")
    item.click()
    expect(page.locator("#answers article.card").first).to_contain_text("March 31")

    page.get_by_role("tab", name="Check a text").click()
    page.locator("#check-text").fill("Full-time employees receive 30 days of paid leave per calendar year.")
    page.get_by_role("button", name="Check", exact=True).click()
    expect(page.locator("#check-result .claim-status").first).to_have_text("Contradicted", timeout=120_000)

    page.get_by_role("button", name="Delete 01_leave_policy.txt").click()
    expect(page.locator("#doc-empty")).to_be_visible()

    page.get_by_role("button", name="Sign out").click()
    expect(page.get_by_role("heading", name="Sign in")).to_be_visible()
    page.get_by_label("Username").fill("bob")
    page.get_by_label("Password").fill("not the password")
    page.locator("#auth-submit").click()
    expect(page.locator("#auth-error")).to_have_text("Incorrect username or password.")
    page.get_by_label("Password").fill(PASSWORD)
    page.locator("#auth-submit").click()
    expect(page.locator("#user-name")).to_have_text("bob")
    page.get_by_role("tab", name="History").click()
    expect(page.locator(".history-item")).to_have_count(1)


def test_session_survives_reload_and_users_are_separate(browser, server):
    first = browser.new_context()
    page = first.new_page()
    register(page, server, "carol")
    upload(page, "07_it_equipment_policy.txt")
    page.reload()
    expect(page.locator("#user-name")).to_have_text("carol")
    expect(page.locator("#doc-list")).to_contain_text("07_it_equipment_policy.txt")

    second = browser.new_context()
    other = second.new_page()
    register(other, server, "dave")
    expect(other.locator("#doc-count")).to_have_text("0")
    first.close()
    second.close()


def test_small_screen_has_no_horizontal_scroll(browser, server):
    context = browser.new_context(viewport={"width": 390, "height": 844})
    page = context.new_page()
    page.goto(server)
    expect(page.locator("#auth-view")).to_be_visible()
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    register(page, server, "erin")
    upload(page, "01_leave_policy.txt")
    ask(page, "How many paid leave days does a full-time employee receive annually?")
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    context.close()


def test_local_mode_adds_a_folder_and_a_webpage(page, server, webpage, tmp_path):
    folder = tmp_path / "policies"
    folder.mkdir()
    (folder / "leave.txt").write_bytes((DOCS / "01_leave_policy.txt").read_bytes())
    register(page, server, "frank")
    expect(page.locator("#storage-note")).to_contain_text("Saved on this computer")

    page.get_by_label("Add from this computer").fill(str(folder))
    page.locator("#path-button").click()
    expect(page.locator("#doc-list")).to_contain_text("leave.txt", timeout=60_000)
    expect(page.locator("#doc-list")).to_contain_text("On this computer")

    page.get_by_label("Add a webpage").fill(webpage)
    page.locator("#url-button").click()
    expect(page.locator("#doc-list")).to_contain_text("Webpage · localhost", timeout=60_000)
    card = ask(page, "Where can employees park?")
    expect(card.locator(".answer-text")).to_contain_text("basement garage")
    expect(card.locator(".source-where").first).to_contain_text(webpage)

    page.get_by_role("button", name="Delete leave.txt").click()
    expect(page.locator("#doc-list")).not_to_contain_text("leave.txt")
    assert (folder / "leave.txt").exists()       # removed from the app, not from the computer

    page.get_by_label("Add from this computer").fill("C:/no/such/folder")
    page.locator("#path-button").click()
    expect(page.locator("#uploads li.error")).to_contain_text("does not exist")


def test_web_mode_deletes_documents_at_sign_out(page, web_server):
    register(page, web_server, "visitor")
    expect(page.locator("#storage-note")).to_contain_text("Signing out deletes")
    expect(page.locator("#path-form")).to_be_hidden()
    upload(page, "01_leave_policy.txt")
    card = ask(page, "How many paid leave days does a full-time employee receive annually?")
    expect(card.locator(".answer-text")).to_contain_text("24 days")

    page.get_by_role("button", name="Sign out").click()          # the confirm prompt is accepted
    expect(page.locator("#auth-fineprint")).to_contain_text("deleted when you sign out")
    page.get_by_label("Username").fill("visitor")
    page.get_by_label("Password").fill(PASSWORD)
    page.locator("#auth-submit").click()
    expect(page.locator("#user-name")).to_have_text("visitor")
    expect(page.locator("#doc-count")).to_have_text("0")
    page.get_by_role("tab", name="History").click()
    expect(page.locator("#history-empty")).to_be_visible()


def test_ai_settings_and_ai_answers(page, server, monkeypatch):
    """Enable a local model in AI settings and ask with 'AI answer' (fake Ollama, fixed reply)."""
    import http.server
    import json

    class FakeOllama(http.server.BaseHTTPRequestHandler):
        def _send(self, payload):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send({"models": [{"name": "fake:1b"}]})

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self._send({"message": {"content": "Full-time employees receive 24 days of paid leave per calendar "
                                               "year. Every employee also gets a free car."}})

        def log_message(self, *args):
            pass

    fake = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=fake.serve_forever, daemon=True).start()
    monkeypatch.setenv("OLLAMA_HOST", f"http://127.0.0.1:{fake.server_address[1]}")
    try:
        register(page, server, "grace")
        upload(page, "01_leave_policy.txt")
        expect(page.locator("#style-switch")).to_be_hidden()          # AI is off by default

        page.get_by_role("button", name="AI settings").click()
        expect(page.locator("#ai-status")).to_contain_text("qwen3:8b is not installed")
        page.locator("#ai-model").select_option("fake:1b")
        page.get_by_label("Offer AI-written answers").check()
        page.locator("#ai-save").click()
        expect(page.locator("#ai-dialog")).to_be_hidden()
        expect(page.locator("#style-ai")).to_have_text("AI answer · fake:1b")

        page.locator("#style-ai").click()
        card = ask(page, "How many paid leave days does a full-time employee receive annually?")
        expect(card.locator(".ai-tag")).to_have_text("AI · fake:1b")
        expect(card.locator(".answer-text")).to_contain_text("24 days")
        expect(card.locator(".answer-text")).not_to_contain_text("free car")
        card.get_by_text("How this answer was checked").click()
        expect(card.locator(".draft")).to_contain_text("free car")       # original shown in details
    finally:
        fake.shutdown()
