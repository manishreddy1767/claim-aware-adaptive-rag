"""``carag-desktop``: the application in its own window, like any installed desktop app.

Two processes:

* this one shows the window (pywebview, Microsoft Edge WebView2 on Windows) and
  nothing else; it never imports PyTorch or loads a model, so the window always
  responds;
* a hidden background process runs the local server and the models
  (``python -m carag.server --no-browser``), with its output in logs/server.log.

Start-up is made robust against the ways it can go wrong on a user's laptop:

* opened twice quickly: only one engine ever runs; a second launch waits for the
  first one's engine and opens a window on it;
* an engine left behind by a crash or a forced close: detected (by its program path)
  and stopped before a new one starts;
* no or slow internet: once the models are installed, the engine loads them offline
  (no update checks against huggingface.co); if that fails it retries online once;
* the engine reports the address it actually uses (``--ready-file``);
* if it cannot start, the window shows the end of the log and where to find it.

Closing the window stops the engine this launch started.
"""

from __future__ import annotations

import html
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
START_TIMEOUT_S = 420
_ERROR_ALREADY_EXISTS = 183

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
<p>Starting… the first start can take a minute.</p><div class="spin"></div></div></body></html>"""

_FAILED = """<!doctype html><html><head><meta charset="utf-8"><style>
:root {{ color-scheme: light dark; }}
body {{ margin: 0; font: 15px 'Segoe UI', sans-serif; padding: 32px; }}
pre {{ background: #8881; padding: 12px; border-radius: 8px; white-space: pre-wrap; font-size: 12px; max-height: 50vh;
      overflow: auto; }}
code {{ background: #8882; padding: 2px 5px; border-radius: 4px; }}
</style></head><body><h2>Claim-Aware RAG could not start</h2>
<p>{reason}</p>
<p>Close this window and open the application again. If it keeps failing, send the file
<code>{log}</code> (or a photo of this screen) to the person who gave you the application.</p>
<p>Last lines of the log:</p><pre>{tail}</pre></body></html>"""


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _healthy(url: str | None) -> bool:
    if not url:
        return False
    try:
        with urllib.request.urlopen(url + "api/health", timeout=2) as response:
            return response.status == 200
    except OSError:
        return False


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _setup_logging(data_dir: Path) -> Path:
    log_dir = data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"
    for name in ("app.log", "server.log"):
        path = log_dir / name
        if path.exists() and path.stat().st_size > 5_000_000:
            path.replace(log_dir / name.replace(".log", ".old.log"))
    logging.basicConfig(level=logging.INFO, filename=log_file, encoding="utf-8",
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return log_file


_MUTEX = None


def _set_windows_identity() -> bool:
    """Taskbar identity and a named mutex (also used by the installer). Returns True when
    another window of this application is already running."""
    global _MUTEX
    if os.name != "nt":
        return False
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
        _MUTEX = ctypes.windll.kernel32.CreateMutexW(None, False, "ClaimAwareRAGDesktop")
        return ctypes.windll.kernel32.GetLastError() == _ERROR_ALREADY_EXISTS
    except Exception:  # pragma: no cover - cosmetic
        return False


def _set_window_icon(window) -> None:
    if os.name != "nt" or not ICON.exists():
        return
    try:
        import clr  # noqa: F401  (pythonnet, used by pywebview on Windows)
        from System.Drawing import Icon  # type: ignore
        window.native.Icon = Icon(str(ICON))
    except Exception:  # pragma: no cover - cosmetic
        logging.getLogger(__name__).debug("Could not set the window icon", exc_info=True)


def _process_path(pid: int) -> str | None:
    """Full program path of a running process, or None (Windows)."""
    if os.name != "nt" or not pid:
        return None
    import ctypes
    from ctypes import wintypes
    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return None
    try:
        buffer = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(1024)
        if ctypes.windll.kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return buffer.value
        return None
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def _stop_leftover_engine(instance: dict, log) -> None:
    """Stop an engine of this installation that is still running but does not answer
    (left behind by a crash or a forced close), so it does not hold memory or the GPU."""
    pid = instance.get("server_pid")
    if _same_installation(pid):
        log.warning("Stopping the engine left from an earlier start (process %s)", pid)
        subprocess.run(["taskkill", "/pid", str(pid), "/t", "/f"], capture_output=True)


def _start_engine(data_dir: Path, offline: bool) -> tuple[subprocess.Popen, Path]:
    ready_file = data_dir / "engine.json"
    ready_file.unlink(missing_ok=True)
    log = open(data_dir / "logs" / "server.log", "a", encoding="utf-8", buffering=1)
    log.write(f"\n--- engine start ({'offline models' if offline else 'online'}) ---\n")
    command = [sys.executable, "-m", "carag.server", "--no-browser", "--port", str(_free_port()),
               "--data-dir", str(data_dir), "--ready-file", str(ready_file)]
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    if offline:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                               creationflags=flags, env=env)
    return process, ready_file


def _log_tail(path: Path, lines: int = 25) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "(no log yet)"
    return "\n".join(text[-lines:])


class Engine:
    """The background engine process this launch started (or the running one it reuses)."""

    def __init__(self, data_dir: Path, log):
        self.data_dir = data_dir
        self.log = log
        self.process: subprocess.Popen | None = None
        self.url: str | None = None
        self.failure: str | None = None

    def start(self, instance_file: Path) -> bool:
        """Start the engine and wait until it answers. Offline first when the models are installed."""
        attempts = [True, False] if (self.data_dir / "models_installed.json").exists() else [False]
        for offline in attempts:
            self.process, ready_file = _start_engine(self.data_dir, offline)
            instance_file.write_text(json.dumps({"url": None, "pid": os.getpid(),
                                                 "server_pid": self.process.pid}), encoding="utf-8")
            self.log.info("Starting the engine (process %s, %s)", self.process.pid,
                          "offline models" if offline else "online")
            deadline = time.time() + START_TIMEOUT_S
            while time.time() < deadline:
                url = _read_json(ready_file).get("url")
                if _healthy(url):
                    self.url = url
                    instance_file.write_text(json.dumps({"url": url, "pid": os.getpid(),
                                                         "server_pid": self.process.pid}), encoding="utf-8")
                    self.log.info("Engine ready at %s", url)
                    return True
                if self.process.poll() is not None:
                    self.log.error("The engine stopped during start-up (exit code %s)", self.process.returncode)
                    break
                time.sleep(0.3)
            else:
                self.log.error("The engine did not answer within %s s", START_TIMEOUT_S)
                self.stop()
                self.failure = "The application's engine took too long to start."
                return False
            self.failure = "The application's engine stopped while starting."
        return False

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()


def _named_mutex(name: str) -> bool:
    """Create (or open) a named mutex; True when it already existed."""
    import ctypes
    handle = ctypes.windll.kernel32.CreateMutexW(None, False, name)
    _HANDLES.append(handle)
    return ctypes.windll.kernel32.GetLastError() == _ERROR_ALREADY_EXISTS


_HANDLES: list = []


def _windows_of(pid: int | None) -> list[int]:
    """Visible top-level application windows owned by a process."""
    if os.name != "nt" or not pid:
        return []
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def callback(hwnd, _):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and user32.IsWindowVisible(hwnd):
            title = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(hwnd, title, 256)
            if title.value == APP_NAME:
                found.append(hwnd)
        return True

    user32.EnumWindows(callback, 0)
    return found


def _bring_to_front(hwnd: int) -> None:
    import ctypes
    user32 = ctypes.windll.user32
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, 9)        # SW_RESTORE
    user32.SetForegroundWindow(hwnd)


def _same_installation(pid: int | None) -> bool:
    """True when the process runs this installation's Python. On Windows a virtual
    environment's python.exe is a small launcher that starts the base interpreter, so
    the process doing the work runs the base Python: both folders count."""
    path = _process_path(pid) if pid else None
    if not path:
        return False
    folders = {Path(sys.executable).resolve().parent,
               Path(getattr(sys, "_base_executable", sys.executable)).resolve().parent}
    return Path(path).resolve().parent in folders


def main(argv: list[str] | None = None) -> int:
    import hashlib
    from .launcher import default_data_dir

    data_dir = default_data_dir()
    log_file = _setup_logging(data_dir)
    _set_windows_identity()
    log = logging.getLogger("carag.desktop")
    instance_file = data_dir / "instance.json"
    window_file = data_dir / "window.json"

    # One window per data folder, like most desktop apps: opening it again brings the open
    # window to the front. A process that holds the marker but shows no window (left over
    # from a crash, or an older version that froze) is stopped, and start-up continues.
    if os.name == "nt":
        key = hashlib.sha1(str(data_dir.resolve()).lower().encode()).hexdigest()[:16]
        if _named_mutex(f"ClaimAwareRAG-{key}"):
            for _ in range(60):                       # up to 15 s for its window to appear
                owner = _read_json(window_file).get("pid")
                windows = _windows_of(owner)
                if windows:
                    _bring_to_front(windows[0])
                    log.info("Already open (process %s); brought its window to the front", owner)
                    return 0
                time.sleep(0.25)
            owner = _read_json(window_file).get("pid")
            if owner and owner != os.getpid() and _same_installation(owner):
                log.warning("Stopping process %s: it holds the application open but shows no window", owner)
                subprocess.run(["taskkill", "/pid", str(owner), "/t", "/f"], capture_output=True)
    window_file.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")

    engine = Engine(data_dir, log)
    instance = _read_json(instance_file)
    adopted_pid = None
    if _healthy(instance.get("url")):
        # An engine whose window is gone (closed abnormally): take it over, stop it on close.
        engine.url = instance["url"]
        adopted_pid = instance.get("server_pid")
        instance_file.write_text(json.dumps({"url": engine.url, "pid": os.getpid(),
                                             "server_pid": adopted_pid}), encoding="utf-8")
        log.info("Reusing the running engine at %s", engine.url)
    else:
        _stop_leftover_engine(instance, log)

    def start_and_show(window) -> None:
        if engine.start(instance_file):
            window.load_url(engine.url)
        else:
            server_log = data_dir / "logs" / "server.log"
            window.load_html(_FAILED.format(reason=html.escape(engine.failure or "Unknown error."),
                                            log=html.escape(str(server_log)),
                                            tail=html.escape(_log_tail(server_log))))

    storage = data_dir / "webview"
    storage.mkdir(parents=True, exist_ok=True)   # created here: pywebview's own creation races
    try:
        import webview
        if engine.url:
            window = webview.create_window(APP_NAME, engine.url, width=1280, height=840, min_size=(420, 560),
                                           text_select=True)
        else:
            window = webview.create_window(APP_NAME, html=_STARTING, width=1280, height=840,
                                           min_size=(420, 560), text_select=True)
            window.events.shown += lambda: threading.Thread(target=start_and_show, args=(window,),
                                                            daemon=True).start()
        window.events.shown += lambda: (_set_window_icon(window), log.info("Window shown"))
        webview.start(private_mode=False, storage_path=str(storage))
        log.info("Window closed")
    except Exception:
        log.exception("Native window unavailable; opening the browser instead")
        import webbrowser
        if engine.url or engine.start(instance_file):
            webbrowser.open(engine.url)
            if engine.process is not None:
                engine.process.wait()

    engine.stop()
    if adopted_pid:
        _stop_leftover_engine({"server_pid": adopted_pid}, log)
    for path in (instance_file, window_file):
        if _read_json(path).get("pid") == os.getpid():
            path.unlink(missing_ok=True)
    (data_dir / "engine.json").unlink(missing_ok=True)
    log.info("Closed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
