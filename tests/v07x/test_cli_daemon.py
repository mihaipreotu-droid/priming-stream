"""Acceptance tests for ``prime daemon start|stop|status|restart`` (spec §2.E).

Most tests stay mock-friendly and exercise the CLI argparse + dispatch
surface without spawning a real daemon. The two gated tests (E1, E3)
launch ``python -m priming_stream.cli.main daemon start --background`` and
verify the round-trip through endpoint file + ``/v1/health``; they are
skipped unless ``RUN_DAEMON_TESTS=1`` to keep the fast loop green
(fastembed cold-load is ~5-10s).

Every test isolates the daemon state dir via
``$PRIMING_STREAM_DAEMON_DIR`` → ``tmp_path``, so the real
``%APPDATA%\\priming-stream`` is never touched.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from priming_stream.cli import daemon as cli_daemon
from priming_stream.cli import main as cli_main
from priming_stream.daemon import lifecycle


_WORKTREE_SRC = Path(__file__).resolve().parents[2] / "src"


def _worktree_env(extra: dict | None = None) -> dict:
    """Build a child-process env that imports ``priming_stream`` from this worktree.

    Without this, a globally installed ``priming_stream`` (the parent repo's
    editable install) would shadow the worktree's modified package.
    """
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    parts = [str(_WORKTREE_SRC)]
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    if extra:
        env.update(extra)
    return env


# ---- shared isolation ---------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_daemon_dir(monkeypatch, tmp_path):
    monkeypatch.setenv(lifecycle.DAEMON_DIR_ENV, str(tmp_path / "daemon"))


# ---- registration / parser surface --------------------------------------


def test_daemon_subcommand_registered_with_four_subcommands():
    """``prime daemon --help`` (via the real parser) advertises the four
    subcommands. We introspect the parser rather than exec'ing a process."""
    parser = cli_main._build_parser()
    # Find the 'daemon' subparser action.
    sub_action = None
    for action in parser._actions:  # noqa: SLF001 — introspecting argparse internals
        if isinstance(action, type(parser._subparsers._group_actions[0])):  # noqa: SLF001
            sub_action = action
            break
    assert sub_action is not None
    assert "daemon" in sub_action.choices

    daemon_parser = sub_action.choices["daemon"]
    # Drill into daemon_parser's own subparsers.
    daemon_sub = None
    for action in daemon_parser._actions:  # noqa: SLF001
        if hasattr(action, "choices") and action.choices and "start" in action.choices:
            daemon_sub = action
            break
    assert daemon_sub is not None
    for name in ("start", "stop", "status", "restart"):
        assert name in daemon_sub.choices, f"missing daemon subcommand: {name}"


def test_daemon_help_runs_clean():
    """``python -m priming_stream.cli.main daemon --help`` exits 0.

    The worktree's ``src/`` may not be the active import location (an
    installed copy of ``priming_stream`` could shadow it). We force it via
    ``PYTHONPATH`` so the test always exercises this worktree's CLI.
    """
    env = _worktree_env()
    proc = subprocess.run(
        [sys.executable, "-m", "priming_stream.cli.main", "daemon", "--help"],
        capture_output=True, text=True, timeout=20, env=env,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr!r}"
    out = proc.stdout
    for name in ("start", "stop", "status", "restart"):
        assert name in out, f"missing in --help: {name}"


# -------------------------------------------------------------------- E4
# status when no endpoint file → "not running", exit 1


def test_e4_status_when_no_daemon(capsys):
    args = argparse_namespace()
    rc = cli_daemon._cmd_status(args)
    assert rc == 1
    captured = capsys.readouterr()
    assert "not running" in captured.out


def test_e4_status_when_endpoint_is_stale(capsys, monkeypatch):
    """Endpoint file present but pid dead → reported as not running."""
    lifecycle.write_endpoint(
        host="127.0.0.1", port=12345, pid=999_999_999,
        started_at="2026-05-27T00:00:00Z",
        version="v0.7-x-bridge-daemon",
    )
    # Force is_endpoint_stale to True regardless of platform pid check.
    monkeypatch.setattr(lifecycle, "is_pid_alive", lambda _pid: False)
    rc = cli_daemon._cmd_status(argparse_namespace())
    assert rc == 1
    assert "not running" in capsys.readouterr().out


# -------------------------------------------------------------------- E2
# stop when no daemon: stderr "not running"; exit 1


def test_e2_stop_when_no_daemon(capsys):
    rc = cli_daemon._cmd_stop(argparse_namespace())
    assert rc == 1
    err = capsys.readouterr().err
    assert "not running" in err


def test_stop_when_endpoint_stale_reports_not_running(capsys, monkeypatch):
    lifecycle.write_endpoint(
        host="127.0.0.1", port=12345, pid=999_999_999,
        started_at="2026-05-27T00:00:00Z",
        version="v0.7-x-bridge-daemon",
    )
    monkeypatch.setattr(lifecycle, "is_pid_alive", lambda _pid: False)
    rc = cli_daemon._cmd_stop(argparse_namespace())
    assert rc == 1
    assert "not running" in capsys.readouterr().err


# ---- mock-server status -------------------------------------------------


class _HealthHandler(BaseHTTPRequestHandler):
    """Minimal ``/v1/health`` responder for status tests without the real
    daemon (no fastembed, no ChromaDB)."""

    def log_message(self, *_a, **_kw):  # silence default access logging
        pass

    def do_GET(self):  # noqa: N802
        if self.path != "/v1/health":
            self.send_response(404)
            self.end_headers()
            return
        body = json.dumps({
            "status": "ok",
            "uptime_s": 42.5,
            "records_count": 7,
            "model_loaded": True,
            "model_name": "stub-model",
            "daemon_version": "v0.7-x-bridge-daemon",
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def mock_health_server(monkeypatch):
    """Spin up a ThreadingHTTPServer that answers /v1/health, and publish
    a matching endpoint file. ``is_pid_alive`` is monkey-patched to True
    so the endpoint isn't classified stale (the server runs in the test's
    own process; its pid is fine, but the explicit patch keeps the test
    deterministic across platforms)."""
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _HealthHandler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        lifecycle.write_endpoint(
            host="127.0.0.1",
            port=srv.server_port,
            pid=os.getpid(),
            started_at="2026-05-27T00:00:00Z",
            version="v0.7-x-bridge-daemon",
        )
        monkeypatch.setattr(lifecycle, "is_pid_alive", lambda _pid: True)
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=2.0)
        lifecycle.remove_endpoint()


def test_status_with_mock_server(mock_health_server, capsys):
    rc = cli_daemon._cmd_status(argparse_namespace())
    assert rc == 0
    out = capsys.readouterr().out
    assert "daemon running on 127.0.0.1" in out
    assert "v0.7-x-bridge-daemon" in out
    assert "uptime_s      42.5" in out
    assert "records_count 7" in out
    assert "model_name    stub-model" in out
    assert "model_loaded  True" in out


def test_status_with_unreachable_endpoint(capsys, monkeypatch):
    """Endpoint file points at a closed port; ``status`` reports unreachable."""
    lifecycle.write_endpoint(
        host="127.0.0.1", port=1,  # privileged + closed
        pid=os.getpid(),
        started_at="2026-05-27T00:00:00Z",
        version="v0.7-x-bridge-daemon",
    )
    monkeypatch.setattr(lifecycle, "is_pid_alive", lambda _pid: True)
    rc = cli_daemon._cmd_status(argparse_namespace())
    assert rc == 1
    err = capsys.readouterr().err
    assert "unreachable" in err or "returned" in err


# ---- start --background with mocked autostart ---------------------------


def test_start_background_already_running(capsys, monkeypatch):
    """If a healthy endpoint exists, ``start`` exits 1 with a message."""
    lifecycle.write_endpoint(
        host="127.0.0.1", port=12345, pid=os.getpid(),
        started_at="2026-05-27T00:00:00Z",
        version="v0.7-x-bridge-daemon",
    )
    monkeypatch.setattr(lifecycle, "is_pid_alive", lambda _pid: True)
    rc = cli_daemon._cmd_start(argparse_namespace(background=True))
    assert rc == 1
    err = capsys.readouterr().err
    assert "already running" in err


def test_start_background_autostart_appears(capsys, monkeypatch):
    """``autostart_daemon`` is stubbed to publish an endpoint file
    immediately; ``start --background`` returns 0 once it sees the file."""

    def _fake_autostart(*, force=False):
        assert force is True, (
            "an explicit `daemon start` must bypass the autostart cooldown"
        )
        lifecycle.write_endpoint(
            host="127.0.0.1", port=23456, pid=os.getpid(),
            started_at="2026-05-27T00:00:00Z",
            version="v0.7-x-bridge-daemon",
        )

    monkeypatch.setattr(lifecycle, "autostart_daemon", _fake_autostart)
    monkeypatch.setattr(lifecycle, "is_pid_alive", lambda _pid: True)
    # Shrink the autostart timeout for fast feedback.
    monkeypatch.setattr(cli_daemon, "_AUTOSTART_TIMEOUT_S", 5.0)
    rc = cli_daemon._cmd_start(argparse_namespace(background=True))
    assert rc == 0
    out = capsys.readouterr().out
    assert "daemon started on 127.0.0.1:23456" in out


def test_start_background_autostart_timeout(capsys, monkeypatch):
    """When autostart never publishes an endpoint, return 1 with timeout msg."""
    monkeypatch.setattr(lifecycle, "autostart_daemon", lambda *, force=False: None)
    monkeypatch.setattr(cli_daemon, "_AUTOSTART_TIMEOUT_S", 1.0)
    rc = cli_daemon._cmd_start(argparse_namespace(background=True))
    assert rc == 1
    err = capsys.readouterr().err
    assert "timed out" in err


# ---- status --all: stray detection (2026-07-16) --------------------------


def _fake_instances(*pids):
    return [
        {"pid": p, "started": "2026-07-16 11:22:56", "ports": [62917],
         "rss_mb": 1500}
        for p in pids
    ]


def test_status_all_flags_a_stray_instance(mock_health_server, capsys, monkeypatch):
    """A daemon that runs, listens and holds memory while owning no endpoint
    is invisible to every client (they only read daemon.json). `--all` must
    surface it and fail, even though the endpoint daemon itself is healthy."""
    monkeypatch.setattr(
        cli_daemon, "_daemon_instances",
        lambda: _fake_instances(os.getpid(), 10328),
    )
    rc = cli_daemon._cmd_status(argparse_namespace(all=True))

    cap = capsys.readouterr()
    assert "pid=10328" in cap.out
    assert "STRAY" in cap.out
    assert "endpoint owner" in cap.out
    assert rc == 1, "a stray alongside a healthy endpoint must not report OK"
    assert "1 daemon instance(s) running outside the endpoint" in cap.err


def test_status_all_is_quiet_when_only_the_endpoint_daemon_runs(
    mock_health_server, capsys, monkeypatch,
):
    monkeypatch.setattr(
        cli_daemon, "_daemon_instances", lambda: _fake_instances(os.getpid()),
    )
    rc = cli_daemon._cmd_status(argparse_namespace(all=True))

    assert rc == 0
    assert "STRAY" not in capsys.readouterr().out


def test_status_without_all_does_not_enumerate_processes(
    mock_health_server, capsys, monkeypatch,
):
    """Plain `status` stays light — no psutil scan, no behaviour change."""
    def _boom():
        raise AssertionError("plain status must not enumerate processes")

    monkeypatch.setattr(cli_daemon, "_daemon_instances", _boom)
    rc = cli_daemon._cmd_status(argparse_namespace())
    assert rc == 0
    assert "instances:" not in capsys.readouterr().out


# ---- argparse_namespace helper ------------------------------------------


def argparse_namespace(**fields):
    """Minimal argparse.Namespace builder for direct cmd-func tests."""
    import argparse as _argparse
    ns = _argparse.Namespace()
    for k, v in fields.items():
        setattr(ns, k, v)
    return ns


# -------------------------------------------------------------------- E1
# Real daemon: start --background, status, then stop. Gated.


@pytest.mark.daemon
@pytest.mark.skipif(
    os.environ.get("RUN_DAEMON_TESTS") != "1",
    reason="set RUN_DAEMON_TESTS=1 to run real-daemon CLI tests",
)
def test_e1_real_start_background_status_stop(tmp_path):
    daemon_dir = tmp_path / "daemon"
    env = _worktree_env({lifecycle.DAEMON_DIR_ENV: str(daemon_dir)})

    # start --background
    p = subprocess.run(
        [sys.executable, "-m", "priming_stream.cli.main", "daemon", "start",
         "--background"],
        capture_output=True, text=True, timeout=60, env=env,
    )
    assert p.returncode == 0, f"start failed: {p.stderr!r}"
    assert "daemon started" in p.stdout

    try:
        # status
        p = subprocess.run(
            [sys.executable, "-m", "priming_stream.cli.main", "daemon", "status"],
            capture_output=True, text=True, timeout=15, env=env,
        )
        assert p.returncode == 0, f"status failed: {p.stderr!r}"
        assert "daemon running" in p.stdout
        assert "uptime_s" in p.stdout
    finally:
        # stop
        p = subprocess.run(
            [sys.executable, "-m", "priming_stream.cli.main", "daemon", "stop"],
            capture_output=True, text=True, timeout=10, env=env,
        )
        # Stop succeeds either via clean cleanup or forced-cleanup branch.
        assert p.returncode == 0, f"stop failed: rc={p.returncode} stderr={p.stderr!r}"

    # status again — should be "not running"
    p = subprocess.run(
        [sys.executable, "-m", "priming_stream.cli.main", "daemon", "status"],
        capture_output=True, text=True, timeout=10, env=env,
    )
    assert p.returncode == 1
    assert "not running" in p.stdout


# -------------------------------------------------------------------- E3
# Real daemon: restart cycle. Gated.


@pytest.mark.daemon
@pytest.mark.skipif(
    os.environ.get("RUN_DAEMON_TESTS") != "1",
    reason="set RUN_DAEMON_TESTS=1 to run real-daemon CLI tests",
)
def test_e3_real_restart_changes_pid(tmp_path):
    daemon_dir = tmp_path / "daemon"
    env = _worktree_env({lifecycle.DAEMON_DIR_ENV: str(daemon_dir)})

    p = subprocess.run(
        [sys.executable, "-m", "priming_stream.cli.main", "daemon", "start",
         "--background"],
        capture_output=True, text=True, timeout=60, env=env,
    )
    assert p.returncode == 0, f"start failed: {p.stderr!r}"

    try:
        info1 = json.loads((daemon_dir / "daemon.json").read_text(encoding="utf-8"))
        pid1 = info1["pid"]
        assert pid1 > 0

        p = subprocess.run(
            [sys.executable, "-m", "priming_stream.cli.main", "daemon", "restart"],
            capture_output=True, text=True, timeout=90, env=env,
        )
        assert p.returncode == 0, f"restart failed: {p.stderr!r}"

        info2 = json.loads((daemon_dir / "daemon.json").read_text(encoding="utf-8"))
        pid2 = info2["pid"]
        assert pid2 > 0
        assert pid2 != pid1, "restart should produce a new pid"
    finally:
        subprocess.run(
            [sys.executable, "-m", "priming_stream.cli.main", "daemon", "stop"],
            capture_output=True, text=True, timeout=10, env=env,
        )
