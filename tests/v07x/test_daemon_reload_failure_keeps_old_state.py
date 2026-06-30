"""R3 — failure during reload leaves the old ``_vec_index`` untouched.
R4 — client helper contract (``reload_daemon`` return / raise behavior).

R3.a (in-process): monkeypatch ``RecordsVecIndex`` to raise on construction;
``POST /v1/reload`` returns 500 with ``{"error": ...}``; module ``_vec_index``
id is unchanged; ``/v1/spread`` continues serving against the old index.

R3.b (gated): real-daemon subprocess test — stdout stays empty even after
a reload failure (file-log only).

R4 covers the client helper itself: silent ``None`` when no endpoint,
parsed dict on 200, raise on 500 / unreachable.
"""
from __future__ import annotations

import http.client
import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from priming_stream.bridge.types import PrimingResult
from priming_stream.daemon import client as daemon_client
from priming_stream.daemon import lifecycle, server


@pytest.fixture(autouse=True)
def _isolated_daemon_dir(monkeypatch, tmp_path):
    monkeypatch.setenv(lifecycle.DAEMON_DIR_ENV, str(tmp_path / "daemon"))


class _StubVecIndex:
    instances: list["_StubVecIndex"] = []

    def __init__(self, persist_dir=None, model_name=None, *, count: int = 5):
        self._count = count
        _StubVecIndex.instances.append(self)

    def count(self) -> int:
        return self._count

    def search(self, query: str, k: int):
        return []


@pytest.fixture
def stub_state(monkeypatch):
    _StubVecIndex.instances.clear()
    initial = _StubVecIndex(count=42)
    monkeypatch.setattr(server, "_vec_index", initial)
    monkeypatch.setattr(server, "_repo", object())
    monkeypatch.setattr(server, "_bridge_cfg", object())
    monkeypatch.setattr(server, "_model_name", "stub-model")
    monkeypatch.setattr(server, "_started_at", "2026-05-27T00:00:00Z")
    monkeypatch.setattr(server, "_started_monotonic", time.monotonic())
    monkeypatch.setattr(server, "_model_warm_ok", True)
    import logging as _logging
    monkeypatch.setattr(server._logger, "handlers", [_logging.NullHandler()])
    return initial


@pytest.fixture
def live_server(stub_state):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), server._Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=2.0)


def _post(srv, path: str, body: dict | None = None) -> tuple[int, dict]:
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_port, timeout=5)
    payload = json.dumps(body or {}).encode("utf-8")
    conn.request(
        "POST", path, body=payload,
        headers={"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, json.loads(data.decode("utf-8"))


# ---- R3.a — reload failure keeps old state -----------------------------


def test_r3a_constructor_failure_returns_500_and_preserves_old_index(
    live_server, monkeypatch, stub_state,
):
    initial_id = id(server._vec_index)

    def _boom(persist_dir, model_name):
        raise RuntimeError("chroma offline")

    monkeypatch.setattr(server, "RecordsVecIndex", _boom)
    status, body = _post(live_server, "/v1/reload", {})
    assert status == 500
    assert "error" in body
    assert "chroma offline" in body["error"]
    # Old index reference unchanged.
    assert id(server._vec_index) == initial_id


def test_r3a_after_failed_reload_spread_still_serves_old_state(
    live_server, monkeypatch, stub_state,
):
    """Spread still works after a failed reload — proves the daemon kept
    its old vec_index reference intact."""
    captured = {}

    def _fake_build_priming(prompt, prev, *, vec_index, repo, conn, cfg,
                            now=None):
        captured["count"] = vec_index.count()
        return PrimingResult(semantic=[], lexical=[])
    monkeypatch.setattr(server, "build_priming", _fake_build_priming)

    # Force reload to fail.
    monkeypatch.setattr(
        server, "RecordsVecIndex",
        lambda persist_dir, model_name: (_ for _ in ()).throw(
            RuntimeError("offline"),
        ),
    )
    status, body = _post(live_server, "/v1/reload", {})
    assert status == 500

    # /v1/spread still serves against the original stub (count=42).
    status, body = _post(live_server, "/v1/spread", {"prompt_text": "x"})
    assert status == 200
    assert captured["count"] == 42


def test_r3a_health_unchanged_after_failed_reload(
    live_server, monkeypatch, stub_state,
):
    monkeypatch.setattr(
        server, "RecordsVecIndex",
        lambda persist_dir, model_name: (_ for _ in ()).throw(
            ValueError("nope"),
        ),
    )
    status, body = _post(live_server, "/v1/reload", {})
    assert status == 500

    conn = http.client.HTTPConnection(
        "127.0.0.1", live_server.server_port, timeout=5,
    )
    conn.request("GET", "/v1/health")
    resp = conn.getresponse()
    health = json.loads(resp.read().decode("utf-8"))
    conn.close()
    assert health["records_count"] == 42
    assert health["model_loaded"] is True  # never touched


# ---- R3.b — real daemon subprocess stays quiet on stdout (gated) -------


@pytest.mark.daemon
@pytest.mark.skipif(
    os.environ.get("RUN_DAEMON_TESTS") != "1",
    reason="set RUN_DAEMON_TESTS=1 to run real-daemon subprocess test",
)
def test_r3b_daemon_subprocess_stdout_quiet_after_reload(tmp_path):
    """Real daemon subprocess: hit /v1/reload (which loads fastembed + Chroma
    if state files exist), assert stdout PIPE remains empty."""
    env = os.environ.copy()
    env[lifecycle.DAEMON_DIR_ENV] = str(tmp_path / "daemon")
    proc = subprocess.Popen(
        [sys.executable, "-m", "priming_stream.daemon.server"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    try:
        # Wait for endpoint file. The daemon needs to finish _init_state
        # (which loads fastembed) before it writes the endpoint. Generous
        # budget since this test runs only under RUN_DAEMON_TESTS=1.
        deadline = time.monotonic() + 30.0
        info = None
        while time.monotonic() < deadline:
            info = lifecycle.read_endpoint()
            if info is not None and not lifecycle.is_endpoint_stale(info):
                break
            time.sleep(0.2)
        assert info is not None and not lifecycle.is_endpoint_stale(info), \
            "daemon did not publish its endpoint within 30s"

        # Issue a reload (may succeed; may fail depending on env state —
        # either way, the binding contract is "no stdout output").
        try:
            result = daemon_client.reload_daemon(timeout_s=15.0)
            assert result is None or isinstance(result, dict)
        except Exception:
            pass
    finally:
        proc.terminate()
        try:
            out, err = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
    assert out == b"", f"daemon wrote to stdout: {out!r}"


# ---- R4.a — reload_daemon returns None when no endpoint ----------------


def test_r4a_returns_none_when_no_endpoint(tmp_path):
    # Autouse fixture already pointed DAEMON_DIR at a tmp subdir with no
    # endpoint file — exactly the "daemon never started" condition.
    assert lifecycle.read_endpoint() is None
    result = daemon_client.reload_daemon(timeout_s=1.0)
    assert result is None


def test_r4a_returns_none_when_endpoint_is_stale():
    """Stale endpoint (pid does not exist) → silent None."""
    lifecycle.write_endpoint(
        host="127.0.0.1",
        port=12345,
        pid=999999,  # very unlikely to exist
        started_at="2026-05-27T00:00:00Z",
        version="v0.7-x-bridge-daemon",
    )
    result = daemon_client.reload_daemon(timeout_s=1.0)
    assert result is None


# ---- R4.b / R4.c / R4.d — client against fake HTTP server --------------


class _CannedHandler(BaseHTTPRequestHandler):
    """Fake daemon: returns a configurable status + body for /v1/reload."""
    status_code: int = 200
    body_obj: dict = {
        "status": "ok",
        "reload_ms": 12.3,
        "records_before": 1,
        "records_after": 2,
        "daemon_version": "v0.7-x-bridge-daemon",
    }

    def log_message(self, format, *args):  # noqa: A002 - silence
        pass

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        if length > 0:
            self.rfile.read(length)
        payload = json.dumps(self.body_obj).encode("utf-8")
        self.send_response(self.status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _spin_fake_daemon(status_code: int, body_obj: dict):
    """Start a fake daemon on a free port; write a real endpoint file."""
    handler = type(
        "_H", (_CannedHandler,),
        {"status_code": status_code, "body_obj": body_obj},
    )
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    lifecycle.write_endpoint(
        host="127.0.0.1",
        port=srv.server_port,
        pid=os.getpid(),
        started_at="2026-05-27T00:00:00Z",
        version="v0.7-x-bridge-daemon",
    )
    return srv, th


def test_r4b_returns_parsed_dict_on_200():
    body = {
        "status": "ok",
        "reload_ms": 84.2,
        "records_before": 155,
        "records_after": 165,
        "daemon_version": "v0.7-x-bridge-daemon",
    }
    srv, th = _spin_fake_daemon(200, body)
    try:
        result = daemon_client.reload_daemon(timeout_s=5.0)
        assert result == body
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=2.0)


def test_r4c_raises_on_500_response():
    body = {"error": "boom"}
    srv, th = _spin_fake_daemon(500, body)
    try:
        with pytest.raises(RuntimeError) as ei:
            daemon_client.reload_daemon(timeout_s=5.0)
        assert "500" in str(ei.value)
        assert "boom" in str(ei.value)
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=2.0)


def test_r4d_raises_on_unreachable_port():
    """Endpoint file points at a port nothing's listening on. Find a free
    port, write the endpoint, do NOT start a server there."""
    # Find a free port then close.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    free_port = s.getsockname()[1]
    s.close()

    lifecycle.write_endpoint(
        host="127.0.0.1",
        port=free_port,
        pid=os.getpid(),  # live pid so is_endpoint_stale → False
        started_at="2026-05-27T00:00:00Z",
        version="v0.7-x-bridge-daemon",
    )
    with pytest.raises((OSError, ConnectionError, socket.timeout)):
        daemon_client.reload_daemon(timeout_s=1.0)


# ---- smoke import surface ----------------------------------------------


def test_client_smoke_import():
    from priming_stream.daemon.client import reload_daemon, spread  # noqa: F401
    assert callable(reload_daemon)
