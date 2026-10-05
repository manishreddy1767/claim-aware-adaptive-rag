"""``carag-desktop``: the application in its own window, like any installed desktop app.

The local server runs in a background thread and the interface is shown in a native
window (pywebview, using Microsoft Edge WebView2 on Windows). There is no console
window; output goes to a log file in the data folder. Closing the window stops the
application. If the application is already running, the new window reuses it.
If no native window can be created, the default browser is opened instead.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

APP_NAME = "Claim-Aware RAG"
APP_ID = "ClaimAwareRAG.Desktop"          # Windows taskbar grouping; shortcuts use the same id
ICON = Path(__file__).resolve().parent / "static" / "icon.ico"


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
    if sys.stdout is None:   # pythonw: no console
        sys.stdout = sys.stderr = open(log_dir / "console.log", "a", encoding="utf-8", buffering=1)
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


def main(argv: list[str] | None = None) -> int:
    from .launcher import default_data_dir
    from .prefetch import quiet_model_libraries

    data_dir = default_data_dir()
    _setup_logging(data_dir)
    quiet_model_libraries()
    _set_windows_identity()
    log = logging.getLogger("carag.desktop")
    instance_file = data_dir / "instance.json"

    # Reuse a running instance instead of loading the models twice.
    server = thread = None
    url = None
    try:
        running = json.loads(instance_file.read_text(encoding="utf-8"))
        if _healthy(running["url"]):
            url = running["url"]
            log.info("Reusing the running instance at %s", url)
    except (OSError, ValueError, KeyError):
        pass

    if url is None:
        import uvicorn
        from .app import Settings, create_app

        port = _free_port()
        url = f"http://127.0.0.1:{port}/"
        app = create_app(Settings(data_dir=data_dir, mode="local"))
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning",
                                               log_config=None))
        thread = threading.Thread(target=server.run, name="server", daemon=True)
        thread.start()
        for _ in range(300):
            if server.started:
                break
            time.sleep(0.05)
        instance_file.write_text(json.dumps({"url": url, "pid": os.getpid()}), encoding="utf-8")
        log.info("Started %s at %s (data: %s)", APP_NAME, url, data_dir)

    try:
        import webview
        window = webview.create_window(APP_NAME, url, width=1280, height=840, min_size=(420, 560),
                                       text_select=True)
        window.events.shown += lambda: _set_window_icon(window)
        webview.start(private_mode=False, storage_path=str(data_dir / "webview"))
    except Exception:
        log.exception("Native window unavailable; opening the browser instead")
        import webbrowser
        webbrowser.open(url)
        if thread is not None:
            thread.join()

    if server is not None:
        server.should_exit = True
        thread.join(timeout=10)
        try:
            if json.loads(instance_file.read_text(encoding="utf-8")).get("pid") == os.getpid():
                instance_file.unlink()
        except (OSError, ValueError):
            pass
        log.info("Closed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
