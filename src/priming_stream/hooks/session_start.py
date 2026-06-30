"""SessionStart hook — log a session-boundary event to the episodic log."""
from __future__ import annotations

import json
import sys

from priming_stream.core.config import Config, load_config
from priming_stream.core.episodic import EpisodicStore
from priming_stream.core.models import now_iso
from priming_stream.core.paths import resolve_paths


def process(event: dict, config: Config) -> dict:
    """Record the session start. Writes only to the episodic log."""
    session_id = str(event.get("session_id", "") or "")
    episodic = EpisodicStore(resolve_paths(config).episodic_dir)
    episodic.append_event({
        "type": "session_start",
        "session_id": session_id,
        "source": str(event.get("source", "") or ""),
        "at": now_iso(),
    })
    return {}


def _read_event() -> dict:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def main() -> None:
    event = _read_event()
    config = load_config()
    try:
        output = process(event, config)
    except Exception as exc:  # hooks must not crash the conversation
        output = {"_bridge_error": str(exc)}
    sys.stdout.write(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
