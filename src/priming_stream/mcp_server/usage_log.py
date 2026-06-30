"""Active-use telemetry writer — appends one ``usage.jsonl`` line per MCP read.

The passive half (what the bridge pushed into a turn) is logged by the hook to
``echoes.jsonl`` (E.1). This is the ACTIVE half: every record-bearing MCP read
the agent makes — a chunk fetch, a follow-up search, a pull-bridge call — is a
deliberate use of the substrate, and the signal Phase-5 calibration needs
("of what was surfaced, what got used; what did the agent search for that
priming missed").

Written by :func:`priming_stream.mcp_server.server.dispatch_tool` AFTER a tool handler
returns successfully. Best-effort and fully swallowed: telemetry must never
break or slow a tool call. ``PRIMING_STREAM_USAGE_OFF`` disables the channel entirely (set
by the test suite; doubles as a kill-switch). 30-day retention, gated prune
(mirrors the echo writer).

Caveat (documented, not hidden): a fetch is a CLEAN but PARTIAL, biased-low
proxy. A record can feed the answer straight from the summary already injected
into context, with no fetch at all (invisible use). This is "verified-use", a
floor — NOT a complete use count.

Schema (one JSON object per line; ``null`` where a field does not apply — role
is derived from ``tool`` at read time, not stored):

    {"at", "session_id", "tool", "record_id", "query", "mode",
     "result_ids", "elapsed_ms"}
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from priming_stream.core.usage_join import (
    FETCH_TOOLS,
    PULL_TOOLS,
    SEARCH_TOOLS,
    SKIP_TOOLS,
)

_USAGE_RETENTION_DAYS = 30
_AT_FMT = "%Y-%m-%dT%H:%M:%SZ"

# Which argument carries the free-text query, per tool. Search tools name it
# ``query_text``; the pull-bridges use their own argument names.
_QUERY_ARG = {
    "graph_search_records": "query_text",
    "graph_search_lexical": "query_text",
    "graph_spread": "text",
    "graph_salient_context": "message",
    "graph_disambiguate": "text",
}


def _parse_at(line: str) -> datetime | None:
    try:
        raw = json.loads(line).get("at", "")
        return datetime.strptime(raw, _AT_FMT).replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _prune(path: Path, cutoff: datetime) -> None:
    """Drop usage lines older than the retention window.

    Gated like the echo prune: only when the FIRST (oldest) line predates the
    cutoff is the O(file) rewrite done, so it runs ~once a day, never per call.
    Atomic via temp + ``os.replace``.
    """
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        first = fh.readline().strip()
    if not first:
        return
    first_at = _parse_at(first)
    if first_at is not None and first_at >= cutoff:
        return
    keep: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            at = _parse_at(ln)
            if at is not None and at >= cutoff:
                keep.append(ln)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        "\n".join(keep) + ("\n" if keep else ""), encoding="utf-8",
    )
    os.replace(tmp, path)


def _result_ids(result) -> list[str] | None:
    """Pull record ids out of a tool result list (search / spread).

    Pull-bridges that return rendered markdown (``graph_salient_context``,
    ``graph_disambiguate``) yield a string — no structured ids — so this
    returns ``None`` for them; the query is still logged.
    """
    if not isinstance(result, list):
        return None
    ids = [
        r.get("record_id")
        for r in result
        if isinstance(r, dict) and r.get("record_id")
    ]
    return ids or None


def _build_line(tool: str, arguments: dict, result, elapsed_ms) -> dict | None:
    """Build the usage record for one tool call, or ``None`` to skip.

    ``graph_stats`` (and any non-record-bearing tool) returns ``None`` — not
    substrate use, just noise for calibration.
    """
    if tool in SKIP_TOOLS:
        return None
    args = arguments if isinstance(arguments, dict) else {}
    record_id = None
    query = None
    mode = None
    result_ids = None
    if tool in FETCH_TOOLS:
        rid = args.get("record_id")
        record_id = str(rid) if rid else None
    elif tool in SEARCH_TOOLS or tool in PULL_TOOLS:
        q = args.get(_QUERY_ARG.get(tool, ""))
        query = str(q) if q else None
        if tool == "graph_search_lexical":
            m = args.get("mode")
            mode = str(m) if m else None
        result_ids = _result_ids(result)
    else:
        # Unknown record-bearing tool — log the bare call so it is not lost.
        pass
    now = datetime.now(timezone.utc)
    return {
        "at": now.strftime(_AT_FMT),
        "session_id": os.environ.get("CLAUDE_CODE_SESSION_ID", ""),
        "tool": tool,
        "record_id": record_id,
        "query": query,
        "mode": mode,
        "result_ids": result_ids,
        "elapsed_ms": (
            round(float(elapsed_ms), 2) if elapsed_ms is not None else None
        ),
    }


def log_usage(
    tool: str,
    arguments: dict,
    result,
    elapsed_ms,
    graph_db: Path,
) -> None:
    """Append one active-use line to ``usage.jsonl``, best-effort.

    ``episodic_dir`` is derived from ``graph_db`` (its parent is the storage
    root) — no config load on the tool hot path. Any failure is swallowed so a
    telemetry hiccup never surfaces to the caller. ``PRIMING_STREAM_USAGE_OFF`` short-
    circuits the whole channel.
    """
    if os.environ.get("PRIMING_STREAM_USAGE_OFF"):
        return
    try:
        line_obj = _build_line(tool, arguments, result, elapsed_ms)
        if line_obj is None:
            return
        episodic_dir = Path(graph_db).parent / "episodic"
        episodic_dir.mkdir(parents=True, exist_ok=True)
        path = episodic_dir / "usage.jsonl"
        now = datetime.now(timezone.utc)
        _prune(path, now - timedelta(days=_USAGE_RETENTION_DAYS))
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line_obj, ensure_ascii=False) + "\n")
    except Exception:
        pass
