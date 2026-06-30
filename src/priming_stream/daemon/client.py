"""Stdlib-only daemon HTTP client for the hot-path hook.

NEVER raises — all errors are swallowed; the function returns ``None`` on
any failure. On missing or stale endpoint files the client triggers a
background autostart (see :func:`priming_stream.daemon.lifecycle.autostart_daemon`)
and returns ``None`` immediately; the next hook fire finds a warm daemon.

Deadlines (binding per brief §2):

* connect: 100ms (TCP handshake budget)
* total:   800ms (hard cap on the whole request)

The hook must not wait longer than that for the daemon under any
circumstance — otherwise it would block the Claude Code turn.
"""
from __future__ import annotations

import http.client
import json
import socket
import time

from priming_stream.daemon import lifecycle


def spread(
    prompt_text: str,
    prev_assistant_text: str = "",
    *,
    session_id: str | None = None,
    deadline_ms: int = 800,
    connect_timeout_ms: int = 100,
) -> dict | None:
    """Call ``POST /v1/spread`` on the resident daemon.

    Returns the parsed response dict on success; ``None`` on missing
    endpoint, stale endpoint, slow daemon, connection refused, non-200
    status, malformed JSON, or any unexpected error. Triggers autostart
    when the endpoint file is missing or refers to a dead pid.
    """
    try:
        info = lifecycle.read_endpoint()
        if lifecycle.is_endpoint_stale(info):
            # Cold or stale → kick off background autostart and bail.
            try:
                lifecycle.autostart_daemon()
            except Exception:
                pass
            return None

        # ``info`` is non-None and non-stale here.
        host = str(info.get("host") or "127.0.0.1")  # type: ignore[union-attr]
        port = int(info.get("port") or 0)  # type: ignore[union-attr]
        if port <= 0:
            return None

        body = json.dumps({
            "prompt_text": prompt_text,
            "prev_assistant_text": prev_assistant_text,
            "session_id": session_id,
        }, ensure_ascii=False).encode("utf-8")

        # Roll a single hard deadline across every socket operation, not
        # a per-op budget. Per-op timeouts would let a drip-feed server
        # spend ``remaining_s`` separately on write, getresponse, and
        # read — up to ~3x the budget. Spec §5.9 caps total at 800ms.
        t_deadline = time.monotonic() + (deadline_ms / 1000.0)

        def _remaining() -> float:
            # 10ms floor so settimeout doesn't get a zero / negative
            # value (which would set non-blocking mode on the socket).
            return max(0.01, t_deadline - time.monotonic())

        conn = http.client.HTTPConnection(
            host, port, timeout=connect_timeout_ms / 1000.0,
        )
        try:
            conn.connect()
            # Save the socket reference now: ``http.client`` clears
            # ``conn.sock`` once ``getresponse()`` hands the socket off to
            # the HTTPResponse, so we'd lose the ability to retighten the
            # timeout before ``resp.read()``.
            sock = conn.sock
            if sock is not None:
                sock.settimeout(_remaining())
            conn.request(
                "POST",
                "/v1/spread",
                body=body,
                headers={"Content-Type": "application/json; charset=utf-8"},
            )
            if sock is not None:
                sock.settimeout(_remaining())
            resp = conn.getresponse()
            if sock is not None:
                sock.settimeout(_remaining())
            raw = resp.read()
            if resp.status != 200:
                return None
            return json.loads(raw.decode("utf-8"))
        finally:
            try:
                conn.close()
            except Exception:
                pass
    except (socket.timeout, ConnectionError, OSError,
            ValueError, json.JSONDecodeError):
        return None
    except Exception:
        # Intentional broad catch: this is the hook's hot path. The whole
        # contract of ``spread`` is "never raise". Any unexpected exception
        # type (e.g. http.client edge cases on Windows) falls through to
        # the lexical fallback rather than crashing the CC turn.
        return None


def reload_daemon(timeout_s: float = 5.0) -> dict | None:
    """POST ``/v1/reload`` to the running daemon.

    Returns:
        Parsed JSON response dict on 200.
        ``None`` if the daemon isn't running (no endpoint file, or stale
        endpoint — pid dead / fields missing).

    Raises:
        ``OSError`` / ``socket.timeout`` / ``ConnectionError`` on transport
        errors. ``json.JSONDecodeError`` on malformed response bodies.
        ``RuntimeError`` on non-200 responses (caller decides what to log).

    Unlike :func:`spread` (which swallows ALL errors for the hot path),
    ``reload_daemon`` is called from a CLI context (``sleep-finalize``)
    where the user can see and react to a failure. Spec §4.2 / §5 #5.
    """
    info = lifecycle.read_endpoint()
    if lifecycle.is_endpoint_stale(info):
        # No daemon to talk to — silent skip (spec §5 #6 sleep-finalize
        # non-fatal contract). We deliberately do NOT trigger autostart
        # here: reload after sleep cycle is the trigger, not a spread call.
        return None

    host = str(info.get("host") or "127.0.0.1")  # type: ignore[union-attr]
    port = int(info.get("port") or 0)  # type: ignore[union-attr]
    if port <= 0:
        return None

    conn = http.client.HTTPConnection(host, port, timeout=timeout_s)
    try:
        conn.request(
            "POST",
            "/v1/reload",
            body=b"{}",
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
        resp = conn.getresponse()
        raw = resp.read()
        if resp.status != 200:
            # Surface server-side error body to the caller's log.
            try:
                err_body = json.loads(raw.decode("utf-8"))
                err_msg = err_body.get("error", raw.decode("utf-8", "replace"))
            except Exception:
                err_msg = raw.decode("utf-8", "replace")
            raise RuntimeError(
                f"reload returned {resp.status}: {err_msg}"
            )
        return json.loads(raw.decode("utf-8"))
    finally:
        try:
            conn.close()
        except Exception:
            pass
