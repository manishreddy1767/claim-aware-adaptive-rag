"""Build the download for end users: dist/ClaimAwareRAG-windows.zip.

The zip holds only what the application needs (the carag package, the
installer and starter scripts, README and licence), inside one ClaimAwareRAG/
folder. Users unzip it and run install.bat; everything then runs on their own
laptop (its GPU or CPU), with documents saved on its disk.

    python scripts/build_release.py
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
NAME = "ClaimAwareRAG-windows.zip"
TOP = "ClaimAwareRAG"

FILES = ["pyproject.toml", "README.md", "LICENSE", "install.bat", "start.bat", "uninstall.bat"]
SKIP_PARTS = {"__pycache__"}
SKIP_SUFFIXES = {".pyc", ".pyo"}


def package_files() -> list[Path]:
    files = [ROOT / f for f in FILES]
    files += sorted(p for p in (ROOT / "carag").rglob("*")
                    if p.is_file() and not SKIP_PARTS & set(p.parts) and p.suffix not in SKIP_SUFFIXES)
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"Missing files: {missing}")
    return files


def main() -> int:
    DIST.mkdir(exist_ok=True)
    target = DIST / NAME
    files = package_files()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            data = path.read_bytes()
            if path.suffix == ".bat":   # cmd.exe needs Windows line endings
                data = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
            zf.writestr(f"{TOP}/{path.relative_to(ROOT).as_posix()}", data)
    size = target.stat().st_size / 1024
    print(f"Built {target} ({len(files)} files, {size:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
