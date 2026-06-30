"""Stop hook — refresh the idle marker that gates the sleep cycle.

The Windows Task Scheduler triggers the sleep cycle on idle. This hook
records ``last_activity_at`` so idle can be measured from the last turn.
It writes a small JSON state file, never the graph.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from priming_stream.core.config import Config, load_config
from priming_stream.core.episodic import EpisodicStore
from priming_stream.core.models import now_iso
from priming_stream.core.paths import resolve_paths


def _activity_path(config: Config) -> Path:
    return resolve_paths(config).episodic_dir / "last_activity.json"


def process(event: dict, config: Config) -> dict:
    """Update the last-activity marker. Writes only episodic-side state."""
    session_id = str(event.get("session_id", "") or "")
    ts = now_iso()

    path = _activity_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(
            {"session_id": session_id, "last_activity_at": ts},
            fh, ensure_ascii=False, indent=2,
        )

    EpisodicStore(resolve_paths(config).episodic_dir).append_event({
        "type": "stop",
        "session_id": session_id,
        "at": ts,
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
