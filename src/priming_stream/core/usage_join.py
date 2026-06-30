"""Active-use telemetry: taxonomy + reader + echo↔usage join (Phase-5 input).

Two complementary telemetry channels live in ``storage/episodic/``:

- ``echoes.jsonl`` — what the bridge PASSIVELY pushed into a turn (written by
  the ``UserPromptSubmit`` hook at prompt-submit; E.1). One line per prompt.
- ``usage.jsonl`` — how the agent ACTIVELY used the substrate (written by the
  MCP server on every record-bearing read tool call; this module's sibling
  :mod:`priming_stream.mcp_server.usage_log`). One line per MCP read.

They are separate files because they are written by separate processes at
separate times (hook dies before any mid-turn fetch happens — the E.1 lesson:
the echo MUST be written in the hook). The *unified per-turn view* the owner
asked for is built here, at READ time: :func:`attach_usage_to_echoes` joins a
usage entry back to the turn that primed it.

Join keys, in priority order (degrades gracefully — never depends on either
alone):

1. ``session_id`` — the MCP subprocess inherits ``CLAUDE_CODE_SESSION_ID`` from
   Claude Code, which equals the hook's echo ``session_id``. Clean key.
2. timestamp + surfaced-set membership — the latest echo at-or-before the
   usage timestamp. Robust fallback for single-user; a fetched id that sits in
   that echo's surfaced set IS the verified-use signal.

Signal classification (verified-use / recall-miss / …) is derived at read time
from the join, not baked into the writer — the writer stays dumb (mirrors the
E.1 design: ids on disk, meaning resolved on read).
"""
from __future__ import annotations

import json
from pathlib import Path

# Tool taxonomy — the single source of truth shared by the writer (what to
# extract / what to skip) and the readers (how to label a usage line). Plain
# data, no MCP imports, so ``core`` stays a leaf below ``mcp_server``.
FETCH_TOOLS = frozenset({"graph_chunk_around_anchor", "graph_records"})
SEARCH_TOOLS = frozenset({"graph_search_records", "graph_search_lexical"})
PULL_TOOLS = frozenset({
    "graph_spread", "graph_salient_context", "graph_disambiguate",
})
# Not record-bearing — a health check, not substrate use. Never logged.
SKIP_TOOLS = frozenset({"graph_stats"})


def role_for(tool: str) -> str:
    """Map a tool name to its active-use role for display/analysis.

    ``fetch`` — pulled a specific record (verification path / metadata look).
    ``search`` — composed a query (semantic or lexical follow-up retrieval).
    ``pull`` — invoked a pull-bridge that returns priming for arbitrary text.
    ``other`` — anything outside the record-bearing surface.
    """
    if tool in FETCH_TOOLS:
        return "fetch"
    if tool in SEARCH_TOOLS:
        return "search"
    if tool in PULL_TOOLS:
        return "pull"
    return "other"


def read_usage(path: Path) -> list[dict]:
    """Tolerant reader for ``usage.jsonl`` (blank/corrupt lines skipped).

    Same posture as the episodic log + ``cli/echoes.py``: a malformed line
    never aborts the read.
    """
    out: list[dict] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                out.append(rec)
    return out


def surfaced_set(echo: dict) -> set[str]:
    """The set of record ids the turn's priming surfaced (semantic ∪ lexical)."""
    return set(echo.get("semantic") or []) | set(echo.get("lexical") or [])


def attach_usage_to_echoes(
    echoes: list[dict], usage: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Attribute each usage entry to the turn that primed it.

    Returns ``(turns, orphans)`` where ``turns`` is ``echoes`` sorted
    chronologically, each augmented with a ``used`` list of the usage entries
    attributed to it, and ``orphans`` is the usage entries that precede every
    echo (no turn to attach to — e.g. a fetch before the first recorded echo).

    The ``at`` field is the fixed ``%Y-%m-%dT%H:%M:%SZ`` format, so lexical
    string comparison is chronological — no datetime parse needed. Join: the
    latest echo at-or-before the usage timestamp, preferring a session match;
    falling back to any session when the usage line carries no ``session_id``
    or no same-session echo precedes it. O(echoes·usage) — fine at telemetry
    volume (hundreds of lines under a 30-day window).
    """
    turns = sorted(echoes, key=lambda e: e.get("at", ""))
    for e in turns:
        e["used"] = []
    orphans: list[dict] = []
    for u in sorted(usage, key=lambda x: x.get("at", "")):
        u_at = u.get("at", "")
        u_sess = u.get("session_id") or ""
        match = None
        if u_sess:
            for e in turns:  # sorted asc → last qualifying is the latest
                if e.get("at", "") <= u_at and (e.get("session_id") or "") == u_sess:
                    match = e
        if match is None:
            for e in turns:
                if e.get("at", "") <= u_at:
                    match = e
        if match is None:
            orphans.append(u)
        else:
            match["used"].append(u)
    return turns, orphans


def classify_usage(u: dict, echo: dict | None) -> str:
    """Label one usage entry against its turn's surfaced set.

    - ``verified-use`` — a fetch of a record this turn's priming surfaced
      (the lower-bound uptake signal: the agent held it to verify the source).
    - ``fetch-unprimed`` — a fetch of a record NOT surfaced this turn (came
      from a prior search / the agent's own memory).
    - ``recall-miss`` — a search/pull that returned ids the priming had NOT
      surfaced (priming should arguably have shown them — a calibration
      blind-spot signal).
    - ``search`` / ``pull`` — a follow-up retrieval whose hits were already
      surfaced (priming covered it).
    """
    role = role_for(u.get("tool", ""))
    surf = surfaced_set(echo) if echo else set()
    if role == "fetch":
        rid = u.get("record_id")
        if rid and rid in surf:
            return "verified-use"
        return "fetch-unprimed"
    if role in ("search", "pull"):
        results = u.get("result_ids") or []
        if results and any(r not in surf for r in results):
            return "recall-miss"
        return role
    return role
