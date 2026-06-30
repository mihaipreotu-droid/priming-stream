"""UserPromptSubmit hook — v0.7-x-bridge-daemon thin shape.

This module is on Claude Code's hot path: a fresh Python process spawns
on every user prompt. Cold-start latency must stay low, so the hook
imports only stdlib plus four stdlib-pure Priming Stream modules: the daemon
HTTP client, the SQLite FTS5 lexical search, the priming markdown
formatter, and the core config/paths layer.

The grep gate in ``tests/v07x/test_hook_thin_imports.py`` enforces this —
no heavyweight identifier (embedding model libraries, ONNX runtime, the
heavyweight bridge layer) may appear anywhere in this file.

Three tiers of behaviour:

1. Daemon warm — ``client.spread`` returns the two priming buckets
   (``semantic`` + ``lexical``) → render via ``render_buckets`` + emit.
2. Daemon cold / slow / errored — ``client.spread`` returns ``None``;
   fall through to FTS5 lexical search on ``records.summary``; render
   with the ``lexical`` source tag. ``client.spread`` also fires the
   detached autostart so the next hook fire finds a warm daemon.
3. Nothing available — emit ``{}`` (no priming, CC proceeds).

Stdout discipline (brief §2): exactly one JSON object on stdout per
invocation. No prints, no logs. All errors swallowed at the boundary —
the hook MUST NOT crash the CC turn.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone

from priming_stream.core.config import load_config
from priming_stream.core.paths import resolve_paths
from priming_stream.daemon import client as daemon_client
from priming_stream.daemon import fallback_lexical
from priming_stream.daemon.render import render_buckets, render_lexical

# E.1 memory echoes — retention window for echoes.jsonl (days).
_ECHO_RETENTION_DAYS = 30
_ECHO_AT_FMT = "%Y-%m-%dT%H:%M:%SZ"


def _parse_echo_at(line: str) -> datetime | None:
    """Parse the ``at`` field of one echo line; None when unparseable."""
    try:
        raw = json.loads(line).get("at", "")
        return datetime.strptime(raw, _ECHO_AT_FMT).replace(
            tzinfo=timezone.utc,
        )
    except Exception:
        return None


def _prune_echoes(path, cutoff: datetime) -> None:
    """Drop echo lines older than the retention window.

    Gate: only when the FIRST line (oldest — appends are chronological)
    predates ``cutoff`` is the file rewritten, so the O(file) rewrite runs
    about once a day, never per-prompt. Unparseable lines are dropped by
    the rewrite. Atomic via temp + ``os.replace``; a concurrent append
    from another session can lose that one line in the rare prune instant
    — acceptable for telemetry.
    """
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        first = fh.readline().strip()
    if not first:
        return
    first_at = _parse_echo_at(first)
    if first_at is not None and first_at >= cutoff:
        return
    keep: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            at = _parse_echo_at(ln)
            if at is not None and at >= cutoff:
                keep.append(ln)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        "\n".join(keep) + ("\n" if keep else ""), encoding="utf-8",
    )
    os.replace(tmp, path)


def _log_echo(
    session_id,
    prompt: str,
    source: str,
    semantic_ids: list,
    lexical_ids: list,
    spread_ms=None,
) -> None:
    """Append one echo line (E.1) to episodic ``echoes.jsonl``, best-effort.

    The echo records what THIS hook invocation actually injected (daemon
    buckets / lexical fallback / nothing) — ids only; summaries stay
    resolvable via SQLite at read time (``prime echoes``). Hooks write
    only to episodic (constitution §working-principles); this is hook
    telemetry, kin to ``live_events.jsonl``. Any failure is swallowed —
    the turn never blocks on the echo. ``PRIMING_STREAM_ECHOES_OFF`` env var disables
    the channel entirely (set by the test suite; doubles as a kill-switch).
    """
    if os.environ.get("PRIMING_STREAM_ECHOES_OFF"):
        return
    try:
        cfg = load_config()
        path = resolve_paths(cfg).episodic_dir / "echoes.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc)
        _prune_echoes(path, now - timedelta(days=_ECHO_RETENTION_DAYS))
        line = json.dumps({
            "at": now.strftime(_ECHO_AT_FMT),
            "session_id": str(session_id or ""),
            "prompt_head": " ".join(prompt.split())[:80],
            "semantic": [str(r) for r in semantic_ids if r],
            "lexical": [str(r) for r in lexical_ids if r],
            "source": source,
            "spread_ms": spread_ms,
        }, ensure_ascii=False)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def _force_utf8_stdio() -> None:
    """Best-effort: switch ``sys.stdout``/``sys.stderr`` to utf-8.

    Windows defaults to cp1252; records routinely carry Romanian diacritics
    + smart quotes that fail to encode there. We prefer ``reconfigure()``
    (Python 3.7+, idempotent, preserves capsys-style captures) over a
    fresh ``TextIOWrapper`` rebind, which would break pytest captures.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8")
        except Exception:
            pass


def _hook_output(text: str) -> str:
    """Wrap rendered priming text in the Claude Code hook envelope."""
    return json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": text,
        }
    }, ensure_ascii=False)


def main() -> None:
    _force_utf8_stdio()
    # Deterministic priming kill-switch (mirrors PRIMING_STREAM_ECHOES_OFF). When
    # set, the hook itself refuses to prime regardless of the settings merge — a
    # robust off-switch independent of Claude Code's `disableAllHooks`.
    # Inert in normal use (env unset).
    if os.environ.get("PRIMING_STREAM_PRIMING_OFF"):
        sys.stdout.write("{}")
        return
    try:
        raw = sys.stdin.read() or "{}"
        event = json.loads(raw)
        if not isinstance(event, dict):
            event = {}
    except Exception:
        sys.stdout.write("{}")
        return

    try:
        prompt = str(event.get("prompt") or "")
        prev = str(event.get("prev_assistant_text") or "")
        session_id = event.get("session_id") or None

        # Tier 1: resident daemon.
        try:
            response = daemon_client.spread(
                prompt, prev, session_id=session_id,
            )
        except Exception:
            response = None

        if isinstance(response, dict):
            semantic = response.get("semantic") or []
            lexical = response.get("lexical") or []
            if semantic or lexical:
                _log_echo(
                    session_id, prompt, "daemon",
                    [it.get("record_id") or it.get("id") for it in semantic],
                    [it.get("record_id") or it.get("id") for it in lexical],
                    response.get("spread_ms"),
                )
                sys.stdout.write(
                    _hook_output(render_buckets(semantic, lexical))
                )
                return

        # Tier 2: lexical fallback.
        try:
            cfg = load_config()
            paths = resolve_paths(cfg)
            hits = fallback_lexical.search(paths.graph_db, prompt, k=10)
        except Exception:
            hits = []
        if hits:
            _log_echo(
                session_id, prompt, "fallback",
                [], [h[0] for h in hits],
            )
            sys.stdout.write(_hook_output(render_lexical(hits)))
            return

        # Tier 3: empty.
        _log_echo(session_id, prompt, "empty", [], [])
        sys.stdout.write("{}")
    except Exception:
        # Final safety net so the hook never crashes the CC turn.
        sys.stdout.write("{}")


if __name__ == "__main__":
    main()
