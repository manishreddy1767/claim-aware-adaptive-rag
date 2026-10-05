"""``carag-desktop``: the application in its own window, like any installed desktop app.

Two processes:

* this one shows the window (pywebview, Microsoft Edge WebView2 on Windows) and
  nothing else; it never imports PyTorch or loads a model, so the window always
  responds;
* a hidden background process runs the local server and the models
  (``python -m carag.server --no-browser``), with its output in the log file.

Loading PyTorch/CUDA in a thread of the window's process could deadlock with
WebView2 starting up (both load system libraries), which froze the window
("Not Responding"); separate processes cannot block each other that way.

The window opens at once with a start-up screen and switches to the application
when the server answers. Closing the window stops the server it started. If the
application is already running, the new window reuses it. If no native window
can be created, the default browser is opened instead.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

APP_NAME = "Claim-Aware RAG"
APP_ID = "ClaimAwareRAG.Desktop"          # Windows taskbar grouping; shortcuts use the same id
ICON = Path(__file__).resolve().parent / "static" / "icon.ico"
START_TIMEOUT_S = 300

_STARTING = """<!doctype html><html><head><meta charset="utf-8"><style>
:root { color-scheme: light dark; }
body { margin: 0; height: 100vh; display: grid; place-items: center; font: 15px 'Segoe UI', system-ui, sans-serif;
       background: #f6f7fb; color: #171a26; }
@media (prefers-color-scheme: dark) { body { background: #0f1117; color: #e8eaf2; } }
.box { text-align: center; }
.mark { width: 48px; height: 48px; border-radius: 12px; background: #4f46e5; margin: 0 auto 18px; position: relative; }
.mark::after { content: ""; position: absolute; left: 16px; top: 9px; width: 12px; height: 22px;
               border: solid #fff; border-width: 0 5px 5px 0; transform: rotate(45deg); }
.spin { width: 22px; height: 22px; border-radius: 50%; border: 3px solid #8885; border-top-color: #4f46e5;
        margin: 16px auto 0; animation: s .8s linear infinite; }
@keyframes s { to { transform: rotate(360deg); } }
p { margin: 6px 0; opacity: .75; }
</style></head><body><div class="box"><div class="mark"></div><strong>Claim-Aware RAG</strong>
<p id="msg">Starting… the first start can take a minute.</p><div class="spin"></div></div></body></html>"""

_FAILED = """<!doctype html><html><head><meta charset="utf-8"><style>
body {{ margin: 0; height: 100vh; display: grid; place-items: center; font: 15px 'Segoe UI', sans-serif; }}
.box {{ max-width: 520px; padding: 24px; }} code {{ background: #8882; padding: 2px 5px; border-radius: 4px; }}
</style></head><body><div class="box"><h2>Claim-Aware RAG could not start</h2>
<p>The application's engine did not start. Details are in the log file:</p><p><code>{log}</code></p>
<p>Close this window and open the application again. If it keeps failing, reinstall it.</p></div></body></html>"""


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(url + "api/health", timeout=2) as response:
            return response.status == 200
    except OSError:
        return False


def _setup_logging(data_dir: Path) -> Path:
    log_dir = data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"
    if log_file.exists() and log_file.stat().st_size > 5_000_000:
        log_file.replace(log_dir / "app.old.log")
    logging.basicConfig(level=logging.INFO, filename=log_file, encoding="utf-8",
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return log_file


_MUTEX = None


def _set_windows_identity() -> None:
    """Group the window under the application's own taskbar icon (not Python's), and hold a
    named mutex so the installer and uninstaller can ask the user to close the application."""
    global _MUTEX
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
            _MUTEX = ctypes.windll.kernel32.CreateMutexW(None, False, "ClaimAwareRAGDesktop")
        except Exception:  # pragma: no cover - cosmetic
            pass


def _set_window_icon(window) -> None:
    if os.name != "nt" or not ICON.exists():
        return
    try:
        import clr  # noqa: F401  (pythonnet, used by pywebview on Windows)
        from System.Drawing import Icon  # type: ignore
        window.native.Icon = Icon(str(ICON))
    except Exception:  # pragma: no cover - cosmetic
        logging.getLogger(__name__).debug("Could not set the window icon", exc_info=True)


def _start_server(data_dir: Path, log_file: Path) -> tuple[subprocess.Popen, str]:
    """Start the server and models in a hidden background process."""
    port = _free_port()
    log = open(log_file.with_name("server.log"), "a", encoding="utf-8", buffering=1)
    command = [sys.executable, "-m", "carag.server", "--no-browser", "--port", str(port),
               "--data-dir", str(data_dir)]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                               creationflags=flags, env=dict(os.environ, PYTHONUNBUFFERED="1"))
    return process, f"http://127.0.0.1:{port}/"


def main(argv: list[str] | None = None) -> int:
    from .launcher import default_data_dir

    data_dir = default_data_dir()
    log_file = _setup_logging(data_dir)
    _set_windows_identity()
    log = logging.getLogger("carag.desktop")
    instance_file = data_dir / "instance.json"

    # Reuse a running instance instead of loading the models twice.
    url, server = None, None
    try:
        running = json.loads(instance_file.read_text(encoding="utf-8"))
        if _healthy(running["url"]):
            url = running["url"]
            log.info("Reusing the running instance at %s", url)
    except (OSError, ValueError, KeyError):
        pass
    if url is None:
        server, url = _start_server(data_dir, log_file)
        instance_file.write_text(json.dumps({"url": url, "pid": os.getpid(), "server_pid": server.pid}),
                                 encoding="utf-8")
        log.info("Starting the engine (process %s) at %s; data: %s", server.pid, url, data_dir)

    def wait_and_show(window) -> None:
        """Switch from the start-up screen to the application once the server answers."""
        deadline = time.time() + START_TIMEOUT_S
        while time.time() < deadline:
            if _healthy(url):
                window.load_url(url)
                log.info("Engine ready")
                return
            if server is not None and server.poll() is not None:
                break
            time.sleep(0.3)
        log.error("The engine did not start (exit code %s); see server.log",
                  server.poll() if server else "n/a")
        window.load_html(_FAILED.format(log=log_file.with_name("server.log")))

    try:
        import webview
        if _healthy(url):
            window = webview.create_window(APP_NAME, url, width=1280, height=840, min_size=(420, 560),
                                           text_select=True)
        else:
            window = webview.create_window(APP_NAME, html=_STARTING, width=1280, height=840,
                                           min_size=(420, 560), text_select=True)
            window.events.shown += lambda: threading.Thread(target=wait_and_show, args=(window,),
                                                            daemon=True).start()
        window.events.shown += lambda: _set_window_icon(window)
        webview.start(private_mode=False, storage_path=str(data_dir / "webview"))
    except Exception:
        log.exception("Native window unavailable; opening the browser instead")
        import webbrowser
        for _ in range(START_TIMEOUT_S * 3):
            if _healthy(url):
                break
            time.sleep(0.3)
        webbrowser.open(url)
        if server is not None:
            server.wait()

    if server is not None:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
        try:
            if json.loads(instance_file.read_text(encoding="utf-8")).get("pid") == os.getpid():
                instance_file.unlink()
        except (OSError, ValueError):
            pass
        log.info("Closed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
