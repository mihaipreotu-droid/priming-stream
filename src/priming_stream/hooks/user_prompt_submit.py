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
import time
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
    client_ms=None,
    prompt_len=None,
    prev_len=None,
    gated=None,
    seed_len=None,
) -> None:
    """Append one echo line (E.1) to episodic ``echoes.jsonl``, best-effort.

    The echo records what THIS hook invocation actually injected (daemon
    buckets / lexical fallback / nothing) — ids only; summaries stay
    resolvable via SQLite at read time (``prime echoes``).

    ``client_ms`` (P7, 2026-07-21) is the hook-side wall time of the daemon
    round-trip, logged on EVERY line — including fallback/empty, where it is
    the only timing left. ``spread_ms`` alone is structurally censored at the
    client deadline (a breach returns ``None`` → no server number reaches the
    echo), which is how the bge long-prompt regression stayed invisible for
    5 days. ``prompt_len`` makes the length-vs-outcome join a field read
    instead of a transcript join. Hooks write
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
            "client_ms": client_ms,
            "prompt_len": prompt_len,
            "prev_len": prev_len,
            "gated": gated,
            "seed_len": seed_len,
        }, ensure_ascii=False)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


# P5 response-seed — the second lineage of the two-seed walk, dormant in
# production until 2026-07-21 because Claude Code never sends
# ``prev_assistant_text`` in the UserPromptSubmit event. The hook now
# recovers it from the tail of the session transcript (``transcript_path``
# IS in the event). Design: prev feeds ONLY the semantic seed (never the
# lexical bucket — lexical terms from the assistant's own reply would just
# echo it); sliced to ~1.2k chars; the outer ``seed_char_budget`` guard
# still applies after the slice. Slice shape chosen EMPIRICALLY
# (2026-07-21): TAIL-ONLY 1200 beat head+tail 400/800 and 200/1000 on a
# replay fixture — more carrier records injected, zero coverage losses
# (the design guess "the head carries the verdict" lost the one moment
# whose support sat in the deep tail). ``head`` stays a parameter
# (0 = tail-only) for recalibration.
_PREV_TAIL_WINDOW_BYTES = 524288  # transcript tail window (512KB)
_PREV_HEAD_CHARS = 0
_PREV_TAIL_CHARS = 1200


def _last_assistant_text(transcript_path) -> str:
    """Most recent assistant reply text from the session transcript (P5).

    Reads the tail of the Claude Code session transcript (JSONL), scans
    backwards for the newest non-sidechain ``type=assistant`` line carrying
    at least one text block, and returns its text blocks joined. Assistant
    entries with only tool_use blocks are skipped (an interrupted turn has
    no reply text). Empty string on missing path, unparseable lines, or any
    error — the turn never blocks on this read.
    """
    try:
        if not transcript_path or not os.path.exists(transcript_path):
            return ""
        with open(transcript_path, "rb") as fh:
            fh.seek(0, 2)
            end = fh.tell()
            fh.seek(max(0, end - _PREV_TAIL_WINDOW_BYTES))
            data = fh.read()
        # utf-8-sig: strips a BOM when the window reaches byte 0 (a BOM'd
        # first line would otherwise fail json.loads); no-op without BOM.
        for ln in reversed(data.decode("utf-8-sig", "replace").splitlines()):
            ln = ln.strip()
            if not ln:
                continue
            try:
                e = json.loads(ln)
            except Exception:
                continue
            if not isinstance(e, dict) or e.get("type") != "assistant":
                continue
            if e.get("isSidechain"):
                continue
            msg = e.get("message") or {}
            content = msg.get("content")
            if isinstance(content, str):
                texts = [content]
            elif isinstance(content, list):
                texts = [b.get("text") or "" for b in content
                         if isinstance(b, dict) and b.get("type") == "text"]
            else:
                texts = []
            text = "\n".join(t for t in texts if t).strip()
            if text:
                return text
        return ""
    except Exception:
        return ""


def _slice_prev(
    text: str,
    head: int = _PREV_HEAD_CHARS,
    tail: int = _PREV_TAIL_CHARS,
) -> str:
    """Slice of the assistant reply (P5 cap, ~1.2k chars).

    Reply shorter than the cap passes through whole. Defaults = tail-only
    1200 (the empirical winner); ``head>0`` re-enables the head+tail shape
    (the rival variant) for recalibration. Applies to EVERY prev — an
    explicitly-passed ``prev_assistant_text`` is respected (not replaced by
    the transcript read) but is still sliced like any other.
    """
    if len(text) <= head + tail:
        return text
    head_part = text[:head].rstrip() + "\n…\n" if head > 0 else ""
    return head_part + text[-tail:].lstrip()


# Turn-gate (2026-07-21): FULL or WHISPER, never silence. The hook computes the regime features — it owns
# the transcript and the echo history — and the daemon applies the policy
# from config. Notification turns WHISPER (execution regime at its extreme;
# a subagent report can hit gotcha-records — the "dirty road" class); their
# seed is TRUNCATABLE at the char budget (the never-truncate rule protects
# only owner-typed text). Sub-floor turns whisper with the weak-field
# marker below. ``cfg.bridge.turn_floor <= 0`` switches the entire gate off.
_NOTIFICATION_PREFIX = "<task-notification>"

# Rendered above the priming block on whisper-floor turns: the epistemic
# signal that silence used to carry, moved into metadata (guard on the
# documented "distant record hijacks the framing" failure mode).
_WEAK_FIELD_MARKER = (
    "⚠ Weak associative field (below the usual relevance threshold) — "
    "treat the associations below as weak suggestions."
)


def _is_notification_turn(prompt: str) -> bool:
    return (prompt or "").lstrip().startswith(_NOTIFICATION_PREFIX)


def _session_turn_index(session_id, cap: int = 10) -> int | None:
    """1-based index of the CURRENT turn in its session, from the echoes tail.

    Counts prior echo lines of this session (capped — the gate only needs to
    know "first few turns or not"). ``None`` on no session id or any error —
    the daemon treats None as non-kickoff-exempt unknown.
    """
    if not session_id:
        return None
    try:
        cfg = load_config()
        path = resolve_paths(cfg).episodic_dir / "echoes.jsonl"
        if not path.exists():
            return 1
        with path.open("rb") as fh:
            fh.seek(0, 2)
            end = fh.tell()
            fh.seek(max(0, end - _DEDUP_TAIL_BYTES))
            data = fh.read()
        sid = str(session_id)
        seen = 0
        for ln in data.decode("utf-8", "replace").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                if str(json.loads(ln).get("session_id") or "") == sid:
                    seen += 1
                    if seen >= cap:
                        break
            except Exception:
                continue
        return seen + 1
    except Exception:
        return None


def _recent_tool_density(transcript_path, window: int = 20) -> float | None:
    """Fraction of tool-using assistant events among the last ``window``
    relevant transcript events (real user prompts / assistant text /
    assistant tool_use) — the execution-regime signal, computed identically
    to the calibration harvest. ``None`` on missing transcript or error.
    """
    try:
        if not transcript_path or not os.path.exists(transcript_path):
            return None
        with open(transcript_path, "rb") as fh:
            fh.seek(0, 2)
            end = fh.tell()
            fh.seek(max(0, end - _PREV_TAIL_WINDOW_BYTES))
            data = fh.read()
        events = []  # "user" | "asst" | "tool"
        for ln in data.decode("utf-8-sig", "replace").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                e = json.loads(ln)
            except Exception:
                continue
            if not isinstance(e, dict) or e.get("isSidechain"):
                continue
            t = e.get("type")
            content = (e.get("message") or {}).get("content")
            if t == "user":
                if isinstance(content, str) and content.strip():
                    events.append("user")
                elif isinstance(content, list):
                    if any(isinstance(b, dict) and b.get("type") == "tool_result"
                           for b in content):
                        continue
                    if any(isinstance(b, dict) and b.get("type") == "text"
                           and (b.get("text") or "").strip() for b in content):
                        events.append("user")
            elif t == "assistant":
                if isinstance(content, list):
                    if any(isinstance(b, dict) and b.get("type") == "text"
                           and (b.get("text") or "").strip() for b in content):
                        events.append("asst")
                    if any(isinstance(b, dict) and b.get("type") == "tool_use"
                           for b in content):
                        events.append("tool")
                elif isinstance(content, str) and content.strip():
                    events.append("asst")
        win = events[-window:]
        if not win:
            return None
        return round(sum(1 for k in win if k == "tool") / len(win), 3)
    except Exception:
        return None


# Item 3.3 cross-turn dedup — bounded tail of echoes.jsonl to reconstruct the
# per-session priming window. 256KB comfortably spans the last N turns of one
# session even when many sessions interleave; measured ~1.5ms/call.
_DEDUP_TAIL_BYTES = 262144


def _recent_primed_ids(session_id) -> list:
    """Union of record ids primed in the last N turns of ``session_id``.

    Source for item 3.3: reads the tail of episodic ``echoes.jsonl``
    (append-only, chronological), scans backward, and unions the ``semantic`` +
    ``lexical`` ids of the most recent ``cfg.bridge.dedup_window_turns`` echoes
    for THIS session. Those ids are handed to the daemon, which drops them
    before truncating each bucket so freed slots backfill from the tail.

    Returns ``[]`` (dedup off this turn) when: no session id, ``PRIMING_STREAM_DEDUP_OFF``
    set, window turns <= 0, the log is absent, or any error — the turn never
    blocks or crashes on the window read. A partial first line from the tail
    seek fails ``json.loads`` and is skipped harmlessly.
    """
    if not session_id or os.environ.get("PRIMING_STREAM_DEDUP_OFF"):
        return []
    try:
        cfg = load_config()
        n = int(getattr(cfg.bridge, "dedup_window_turns", 0) or 0)
        if n <= 0:
            return []
        path = resolve_paths(cfg).episodic_dir / "echoes.jsonl"
        if not path.exists():
            return []
        with path.open("rb") as fh:
            fh.seek(0, 2)
            end = fh.tell()
            fh.seek(max(0, end - _DEDUP_TAIL_BYTES))
            data = fh.read()
        sid = str(session_id)
        ids: set[str] = set()
        seen = 0
        for ln in reversed(data.decode("utf-8", "replace").splitlines()):
            ln = ln.strip()
            if not ln:
                continue
            try:
                e = json.loads(ln)
            except Exception:
                continue
            if str(e.get("session_id") or "") != sid:
                continue
            for r in (e.get("semantic") or []):
                if r:
                    ids.add(str(r))
            for r in (e.get("lexical") or []):
                if r:
                    ids.add(str(r))
            seen += 1
            if seen >= n:
                break
        return list(ids)
    except Exception:
        return []


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

        try:
            _bcfg = load_config().bridge
        except Exception:
            _bcfg = None
        floor = float(getattr(_bcfg, "turn_floor", 0.0) or 0.0)

        # P2/P3: notification turns whisper (flag sent to the daemon, which
        # gates only while ITS turn_floor > 0). The seed truncation below is
        # a LATENCY guard, deliberately independent of the gate: turn_floor=0
        # (gate rollback) must not re-open the giant-notification-embed
        # deadline breach (2026-07-21 review;.
        is_notif = _is_notification_turn(prompt)

        # P5 response-seed: Claude Code never populates prev_assistant_text,
        # so recover the last assistant reply from the session transcript and
        # slice it. An explicitly-passed prev (tests, future CC versions) is
        # respected as-is upstream of the slice.
        if not prev:
            prev = _last_assistant_text(event.get("transcript_path"))
        if prev:
            prev = _slice_prev(prev)

        # Seed char budget (user-first; 2026-07-21). The user
        # prompt is NEVER truncated — it is the turn's intent. ``prev`` (the
        # P5 response-seed, when populated) takes only what remains of the
        # total budget, so the embed stays under the client deadline. Budget
        # <= 0 disables the cap entirely (prev passes through untouched).
        budget = int(getattr(_bcfg, "seed_char_budget", 0) or 0)
        # The never-truncate rule protects owner-typed text only: a
        # notification seed (subagent report, possibly tens of KB) is cut at
        # the budget so the embed stays under the client deadline.
        seed_prompt = prompt[:budget] if (is_notif and budget > 0) else prompt
        if prev and budget > 0:
            prev = prev[: max(0, budget - len(seed_prompt))]

        # P2/P3 regime features for the daemon-side gate (whisper triggers /
        # kickoff). Computed only when the gate is on; a request without
        # features is never gated.
        turn_idx = _session_turn_index(session_id) if floor > 0 else None
        tool_density = (_recent_tool_density(event.get("transcript_path"))
                        if floor > 0 else None)

        # Item 3.3: ids primed in the last N turns of this session — suppressed
        # this turn so the freed slots surface fresh (distal) records instead of
        # re-injecting what the model already saw. Best-effort; [] disables.
        recent_ids = _recent_primed_ids(session_id)

        # Tier 1: resident daemon. Wall time measured hook-side (P7): unlike
        # ``spread_ms`` it survives a deadline breach, so the echo stream
        # keeps an uncensored latency distribution.
        t_client = time.monotonic()
        try:
            response = daemon_client.spread(
                seed_prompt, prev, session_id=session_id,
                recent_ids=recent_ids,
                turn_idx=turn_idx, tool_density=tool_density,
                notification=(True if is_notif else None),
            )
        except Exception:
            response = None
        client_ms = round((time.monotonic() - t_client) * 1000.0, 1)

        daemon_gated = None
        if isinstance(response, dict):
            semantic = response.get("semantic") or []
            lexical = response.get("lexical") or []
            gated = response.get("gated")
            daemon_gated = gated
            if semantic or lexical:
                _log_echo(
                    session_id, prompt, "daemon",
                    [it.get("record_id") or it.get("id") for it in semantic],
                    [it.get("record_id") or it.get("id") for it in lexical],
                    response.get("spread_ms"),
                    client_ms=client_ms,
                    prompt_len=len(prompt),
                    prev_len=len(prev),
                    gated=gated,
                    seed_len=len(seed_prompt),
                )
                text = render_buckets(semantic, lexical)
                if gated == "whisper-floor":
                    # The epistemic signal silence used to carry, as metadata.
                    text = _WEAK_FIELD_MARKER + "\n\n" + text
                sys.stdout.write(_hook_output(text))
                return

        # Tier 2: lexical fallback. Reached on daemon breach/None AND on a
        # daemon 200 with empty buckets — the echo's ``gated`` distinguishes
        # them ("fallback" vs "empty-<verdict>", 2026-07-21 review:, and a
        # whisper-class daemon verdict keeps its cap + marker here instead of
        # silently widening to k=10 unmarked (2026-07-21 review;.
        is_whisper_turn = bool(daemon_gated) and str(daemon_gated).startswith("whisper")
        fallback_k = 10
        if is_whisper_turn:
            fallback_k = int(getattr(_bcfg, "whisper_lex_k", 3) or 3)
        try:
            cfg = load_config()
            paths = resolve_paths(cfg)
            hits = fallback_lexical.search(paths.graph_db, prompt, k=10)
            # Item 3.3: apply the same cross-turn dedup on the cold path —
            # drop recently-primed ids (same-or-fewer, never padded).
            if recent_ids and hits:
                recent_set = set(recent_ids)
                hits = [h for h in hits if h[0] not in recent_set]
            hits = hits[:fallback_k]
        except Exception:
            hits = []
        gated_field = f"empty-{daemon_gated}" if daemon_gated else "fallback"
        if hits:
            _log_echo(
                session_id, prompt, "fallback",
                [], [h[0] for h in hits],
                client_ms=client_ms,
                prompt_len=len(prompt),
                prev_len=len(prev),
                gated=gated_field,
                seed_len=len(seed_prompt),
            )
            text = render_lexical(hits)
            if daemon_gated == "whisper-floor":
                text = _WEAK_FIELD_MARKER + "\n\n" + text
            sys.stdout.write(_hook_output(text))
            return

        # Tier 3: empty.
        _log_echo(session_id, prompt, "empty", [], [],
                  client_ms=client_ms, prompt_len=len(prompt),
                  prev_len=len(prev),
                  gated=(gated_field if daemon_gated else None),
                  seed_len=len(seed_prompt))
        sys.stdout.write("{}")
    except Exception:
        # Final safety net so the hook never crashes the CC turn.
        sys.stdout.write("{}")


if __name__ == "__main__":
    main()
