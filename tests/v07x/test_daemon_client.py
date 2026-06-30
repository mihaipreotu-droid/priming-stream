"""Daemon HTTP client acceptance tests (spec §D1-D3).

These tests use ``tmp_path`` as the daemon state dir (via
``$PRIMING_STREAM_DAEMON_DIR``) so they never touch the developer's
real ``%APPDATA%\\priming-stream``.

D2 / D3 spin up a tiny in-process ``ThreadingHTTPServer`` to exercise the
slow / 500 / connection-refused branches without depending on the real
daemon server module (which would pull fastembed).
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from priming_stream.daemon import client as daemon_client
from priming_stream.daemon import lifecycle


@pytest.fixture(autouse=True)
def _isolated_daemon_dir(monkeypatch, tmp_path):
    monkeypatch.setenv(lifecycle.DAEMON_DIR_ENV, str(tmp_path))


# ---------------------------------------------------------------- helpers


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def _write_self_endpoint(port: int) -> None:
    """Endpoint file pointing at ``127.0.0.1:port`` with current pid (alive)."""
    lifecycle.write_endpoint(
        host="127.0.0.1",
        port=port,
        pid=os.getpid(),
        started_at="2026-05-27T00:00:00Z",
        version="v0.7-x-bridge-daemon",
    )


class _SlowHandler(BaseHTTPRequestHandler):
    """Sleeps long enough to exceed the client deadline before responding."""

    sleep_s = 1.2

    def log_message(self, format, *args):  # noqa: A002
        return

    def do_POST(self):  # noqa: N802
        time.sleep(self.sleep_s)
        try:
            payload = json.dumps({"records": [], "spread_ms": 0.0,
                                  "daemon_version": "v0.7-x-bridge-daemon"})
            data = payload.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception:
            pass


class _DripHandler(BaseHTTPRequestHandler):
    """Sleeps once after reading the request, then again before sending
    the body — exercises the per-socket-op vs total-budget distinction
    (M-1). With a per-op timeout the client could spend ``remaining_s``
    on each leg separately; with a single rolling deadline it can't.
    """

    pre_response_sleep_s = 0.6
    pre_body_sleep_s = 0.6

    def log_message(self, format, *args):  # noqa: A002
        return

    def do_POST(self):  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length > 0:
                self.rfile.read(length)
            time.sleep(self.pre_response_sleep_s)
            payload = json.dumps({"records": [], "spread_ms": 0.0,
                                  "daemon_version": "v0.7-x-bridge-daemon"})
            data = payload.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            # Force the status + headers onto the wire so the client's
            # getresponse() call returns BEFORE we sleep again. Without
            # this flush, send_header collects into a buffer that's not
            # released until the first wfile.write(data) — collapsing
            # the two sleeps into one wire stall, defeating the test.
            try:
                self.wfile.flush()
            except Exception:
                pass
            time.sleep(self.pre_body_sleep_s)
            try:
                self.wfile.write(data)
            except Exception:
                pass
        except Exception:
            pass


class _ErrorHandler(BaseHTTPRequestHandler):
    """Always returns HTTP 500."""

    def log_message(self, format, *args):  # noqa: A002
        return

    def do_POST(self):  # noqa: N802
        data = json.dumps({"error": "boom"}).encode("utf-8")
        self.send_response(500)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class _OkHandler(BaseHTTPRequestHandler):
    """Returns a canned 200 with one record."""

    def log_message(self, format, *args):  # noqa: A002
        return

    def do_POST(self):  # noqa: N802
        data = json.dumps({
            "records": [{"record_id": "rec_a", "summary": "alpha",
                          "rank": 1, "source_uri": "qmd://x",
                          "anchor_start": 0, "anchor_end": 0}],
            "spread_ms": 12.3,
            "daemon_version": "v0.7-x-bridge-daemon",
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _spawn_server(handler_cls):
    """Spawn ``handler_cls`` on a free port; return (server, port, thread)."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    port = server.server_port
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, port, t


# --------------------------------------------------------------- D1 tests


def test_d1_missing_endpoint_returns_none_and_triggers_autostart(monkeypatch):
    """No endpoint file → spread returns None + autostart is invoked."""
    called = {"n": 0}

    def fake_autostart():
        called["n"] += 1

    monkeypatch.setattr(lifecycle, "autostart_daemon", fake_autostart)

    out = daemon_client.spread("hello")
    assert out is None
    assert called["n"] == 1


def test_d1_stale_endpoint_returns_none_and_triggers_autostart(monkeypatch):
    """Endpoint exists but pid is dead → autostart + None."""
    lifecycle.write_endpoint(
        host="127.0.0.1",
        port=1,
        pid=99_999_999,  # overwhelmingly likely to be dead
        started_at="x",
        version="v0.7-x-bridge-daemon",
    )
    called = {"n": 0}

    def fake_autostart():
        called["n"] += 1

    monkeypatch.setattr(lifecycle, "autostart_daemon", fake_autostart)

    out = daemon_client.spread("hello")
    assert out is None
    assert called["n"] == 1


def test_d1_autostart_failure_does_not_raise(monkeypatch):
    """If autostart itself errors, the client still swallows + returns None."""

    def boom():
        raise RuntimeError("popen failed")

    monkeypatch.setattr(lifecycle, "autostart_daemon", boom)
    out = daemon_client.spread("hello")
    assert out is None


# --------------------------------------------------------------- D2 tests


def test_d2_slow_server_aborts_within_deadline():
    """Server sleeping >800ms must return None and not block past budget."""
    server, port, _ = _spawn_server(_SlowHandler)
    try:
        _write_self_endpoint(port)
        t0 = time.monotonic()
        out = daemon_client.spread("hello", deadline_ms=400,
                                    connect_timeout_ms=100)
        elapsed = time.monotonic() - t0
    finally:
        server.shutdown()
        server.server_close()

    assert out is None
    # Generous margin: deadline 400ms + ~few hundred ms slack for thread
    # scheduling on Windows.
    assert elapsed < 1.5, f"client took {elapsed:.2f}s > deadline"


def test_d2_dripfeed_server_aborts_within_total_budget():
    """A server that sleeps before response AND before body (each sleep
    well under the budget, but their sum well over) must still trip the
    rolling-deadline cap. Without M-1 the per-socket-op timeout would
    reset for each leg and the request could span ~3x the budget.
    """
    server, port, _ = _spawn_server(_DripHandler)
    try:
        _write_self_endpoint(port)
        t0 = time.monotonic()
        out = daemon_client.spread("hello", deadline_ms=800,
                                    connect_timeout_ms=100)
        elapsed = time.monotonic() - t0
    finally:
        server.shutdown()
        server.server_close()

    assert out is None
    # Total drip-feed = 0.6 + 0.6 = 1.2s > 800ms budget; with rolling
    # deadline we must abort under ~1.1s (deadline + Windows slack).
    assert elapsed < 1.1, f"client took {elapsed:.2f}s > rolling deadline"


# --------------------------------------------------------------- D3 tests


def test_d3_server_500_returns_none():
    """Daemon error response (5xx) → client returns None."""
    server, port, _ = _spawn_server(_ErrorHandler)
    try:
        _write_self_endpoint(port)
        out = daemon_client.spread("hello")
    finally:
        server.shutdown()
        server.server_close()
    assert out is None


def test_d3_connection_refused_returns_none(monkeypatch):
    """Endpoint file points at a closed port; pid is the test's own (alive)
    so is_endpoint_stale returns False and we try to connect → refused.
    """
    closed_port = _free_port()  # no server bound here
    _write_self_endpoint(closed_port)
    out = daemon_client.spread("hello")
    assert out is None


# -------------------------------------------------------- happy-path smoke


def test_client_happy_path_returns_parsed_dict():
    """Sanity: when the server returns 200 JSON, client parses + returns it."""
    server, port, _ = _spawn_server(_OkHandler)
    try:
        _write_self_endpoint(port)
        out = daemon_client.spread("hello")
    finally:
        server.shutdown()
        server.server_close()
    assert isinstance(out, dict)
    assert out.get("daemon_version") == "v0.7-x-bridge-daemon"
    assert out.get("records") and out["records"][0]["record_id"] == "rec_a"
