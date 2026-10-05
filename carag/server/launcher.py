"""``carag-app``: start the local web application and open it in the browser."""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

DEFAULT_PORT = 8765


def default_data_dir() -> Path:
    """Per-user application data folder (accounts, documents, history); CARAG_DATA_DIR overrides it."""
    if os.environ.get("CARAG_DATA_DIR"):
        return Path(os.environ["CARAG_DATA_DIR"])
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return base / "ClaimAwareRAG"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "ClaimAwareRAG"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "claim-aware-rag"


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex((host, port)) != 0


def _pick_port(host: str, preferred: int) -> int:
    for port in range(preferred, preferred + 20):
        if _port_free(host, port):
            return port
    raise SystemExit(f"No free port between {preferred} and {preferred + 19}; use --port.")


def _open_when_ready(url: str) -> None:
    for _ in range(120):
        try:
            with urllib.request.urlopen(url + "api/health", timeout=2):
                webbrowser.open(url)
                return
        except OSError:
            time.sleep(0.5)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="carag-app", description="Claim-Aware RAG local web application.")
    parser.add_argument("--mode", choices=["local", "web"], default="local",
                        help="local (default): documents are saved on this computer and files can be added "
                             "by path; web: a hosted website, documents stay in memory and are deleted at sign-out")
    parser.add_argument("--host", default="127.0.0.1",
                        help="interface to listen on (default 127.0.0.1: this computer only)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"port (default {DEFAULT_PORT}; the next free one is used if it is busy)")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir(),
                        help=f"where accounts and documents are stored (default {default_data_dir()})")
    parser.add_argument("--no-signup", action="store_true",
                        help="only allow creating the first account; later accounts cannot be created")
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from .prefetch import quiet_model_libraries
    quiet_model_libraries()
    logging.getLogger("httpx").setLevel(logging.WARNING)

    import uvicorn
    from .app import Settings, create_app

    port = _pick_port(args.host, args.port)
    shown_host = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
    url = f"http://{shown_host}:{port}/"
    app = create_app(Settings(data_dir=args.data_dir, mode=args.mode, allow_signup=not args.no_signup))
    print(f"\n  Claim-Aware RAG is running at {url}  ({args.mode} mode)")
    if args.mode == "web":
        print(f"  Accounts are stored in {args.data_dir}; documents and questions are kept in memory only")
        print("  and deleted when a user signs out (or after 2 hours without activity).")
    else:
        print(f"  Data folder: {args.data_dir}")
    if args.host not in ("127.0.0.1", "localhost"):
        print("  Warning: listening beyond this computer; anyone who can reach it can try to sign in.")
    print("  Models load in the background; the first start may download them (about 1 GB).")
    print("  Press Ctrl+C to stop.\n")
    if not args.no_browser:
        threading.Thread(target=_open_when_ready, args=(url,), daemon=True).start()
    uvicorn.run(app, host=args.host, port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
