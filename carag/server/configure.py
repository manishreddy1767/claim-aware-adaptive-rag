"""Configure the installed application from the command line (used by the installer).

    python -m carag.server.configure --ai-model qwen3:8b     # turn AI answers on with this Ollama model
    python -m carag.server.configure --no-ai                 # turn them off
"""

from __future__ import annotations

import argparse
import json
import sys

from .launcher import default_data_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m carag.server.configure")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ai-model", help="enable AI-written answers with this Ollama model")
    group.add_argument("--no-ai", action="store_true", help="disable AI-written answers")
    args = parser.parse_args(argv)

    path = default_data_dir() / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        settings = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        settings = {}
    ai = settings.get("ai", {})
    if args.no_ai:
        ai["enabled"] = False
    else:
        ai.update(enabled=True, model=args.ai_model)
    settings["ai"] = ai
    path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(f"AI answers {'on with ' + ai['model'] if ai.get('enabled') else 'off'} ({path})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
