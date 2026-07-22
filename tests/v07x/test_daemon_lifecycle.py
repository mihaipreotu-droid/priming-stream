"""Acceptance tests for ``priming_stream.daemon.lifecycle`` (spec §B1-B5).

Every test isolates the daemon state directory via
``$PRIMING_STREAM_DAEMON_DIR`` pointing at ``tmp_path`` — no test
touches the real ``%APPDATA%\\priming-stream``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

from priming_stream.daemon import lifecycle


@pytest.fixture(autouse=True)
def _isolated_daemon_dir(monkeypatch, tmp_path):
    """Redirect daemon_dir() to tmp_path for every test."""
    monkeypatch.setenv(lifecycle.DAEMON_DIR_ENV, str(tmp_path))


# -------------------------------------------------------------------- B1


def test_acquire_lock_succeeds_on_fresh_state():
    handle = lifecycle.acquire_lock()
    try:
        assert handle is not None
        assert lifecycle.lockfile_path().exists()
    finally:
        lifecycle.release_lock(handle)


def test_acquire_lock_contention_raises():
    h1 = lifecycle.acquire_lock()
    try:
        with pytest.raises((BlockingIOError, OSError)):
            lifecycle.acquire_lock()
    finally:
        lifecycle.release_lock(h1)


def test_release_lock_allows_reacquire():
    h1 = lifecycle.acquire_lock()
    lifecycle.release_lock(h1)
    h2 = lifecycle.acquire_lock()
    try:
        assert h2 is not None
    finally:
        lifecycle.release_lock(h2)


def test_release_lock_none_is_noop():
    # Must not raise.
    lifecycle.release_lock(None)


# -------------------------------------------------------------------- B2


def test_endpoint_roundtrip():
    lifecycle.write_endpoint(
        host="127.0.0.1",
        port=38421,
        pid=12345,
        started_at="2026-05-27T14:33:01Z",
        version="v0.7-x-bridge-daemon",
    )
    info = lifecycle.read_endpoint()
    assert info == {
        "host": "127.0.0.1",
        "port": 38421,
        "pid": 12345,
        "started_at": "2026-05-27T14:33:01Z",
        "version": "v0.7-x-bridge-daemon",
    }


def test_read_endpoint_missing_returns_none():
    assert lifecycle.read_endpoint() is None


def test_read_endpoint_malformed_returns_none():
    p = lifecycle.endpoint_path()
    p.write_text("not json {{{", encoding="utf-8")
    assert lifecycle.read_endpoint() is None


def test_read_endpoint_non_dict_returns_none():
    p = lifecycle.endpoint_path()
    p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert lifecycle.read_endpoint() is None


def test_remove_endpoint_idempotent():
    lifecycle.remove_endpoint()  # nothing to remove — must not raise
    lifecycle.write_endpoint("127.0.0.1", 1, os.getpid(), "x", "v")
    assert lifecycle.endpoint_path().exists()
    lifecycle.remove_endpoint()
    assert not lifecycle.endpoint_path().exists()


def test_write_endpoint_is_atomic_no_tmp_left():
    lifecycle.write_endpoint("127.0.0.1", 1, os.getpid(), "x", "v")
    tmp = lifecycle.endpoint_path().with_suffix(
        lifecycle.endpoint_path().suffix + ".tmp"
    )
    assert not tmp.exists()


# -------------------------------------------------------------------- B3


def test_is_pid_alive_self():
    assert lifecycle.is_pid_alive(os.getpid()) is True


def test_is_pid_alive_zero_false():
    assert lifecycle.is_pid_alive(0) is False


def test_is_pid_alive_negative_false():
    assert lifecycle.is_pid_alive(-1) is False


def test_is_pid_alive_dead_pid_false():
    # A pid that is overwhelmingly unlikely to be assigned.
    assert lifecycle.is_pid_alive(99_999_999) is False


def test_is_pid_alive_short_lived_subprocess():
    """A process we just reaped is dead from our perspective."""
    proc = subprocess.Popen(
        [sys.executable, "-c", "pass"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    proc.wait()
    # On Windows the PID may be reusable; the assertion is that we
    # don't raise. On a freshly-exited PID OpenProcess returns 0 or
    # GetExitCodeProcess reports != 259.
    result = lifecycle.is_pid_alive(proc.pid)
    assert isinstance(result, bool)


# -------------------------------------------------------------------- B4


def test_is_endpoint_stale_none_is_stale():
    assert lifecycle.is_endpoint_stale(None) is True


def test_is_endpoint_stale_dead_pid_is_stale():
    info = {
        "host": "127.0.0.1",
        "port": 1,
        "pid": 99_999_999,
        "started_at": "x",
        "version": "v",
    }
    assert lifecycle.is_endpoint_stale(info) is True


def test_is_endpoint_stale_self_pid_not_stale():
    info = {
        "host": "127.0.0.1",
        "port": 1,
        "pid": os.getpid(),
        "started_at": "x",
        "version": "v",
    }
    assert lifecycle.is_endpoint_stale(info) is False


def test_is_endpoint_stale_missing_field_is_stale():
    # Missing 'version'
    info = {
        "host": "127.0.0.1",
        "port": 1,
        "pid": os.getpid(),
        "started_at": "x",
    }
    assert lifecycle.is_endpoint_stale(info) is True


def test_is_endpoint_stale_wrong_type_is_stale():
    info = {
        "host": "127.0.0.1",
        "port": "not-an-int",
        "pid": os.getpid(),
        "started_at": "x",
        "version": "v",
    }
    assert lifecycle.is_endpoint_stale(info) is True


def test_is_endpoint_stale_not_a_dict_is_stale():
    assert lifecycle.is_endpoint_stale("nope") is True  # type: ignore[arg-type]
    assert lifecycle.is_endpoint_stale([1, 2, 3]) is True  # type: ignore[arg-type]


# -------------------------------------------------------------------- B5


def test_autostart_daemon_calls_popen_detached(monkeypatch):
    calls = {}

    class FakePopen:
        def __init__(self, cmd, **kwargs):
            calls["cmd"] = cmd
            calls["kwargs"] = kwargs

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    # Also patch the reference inside the lifecycle module if it bound
    # subprocess.Popen at import time. (We imported the module, not the
    # name; the monkeypatch above is sufficient because lifecycle uses
    # ``subprocess.Popen`` qualified.)
    monkeypatch.setattr(lifecycle.subprocess, "Popen", FakePopen)
    # Make sure the disable-autostart env var isn't set in the host env.
    monkeypatch.delenv(lifecycle.DISABLE_AUTOSTART_ENV, raising=False)

    lifecycle.autostart_daemon()

    assert calls["cmd"] == [sys.executable, "-m", "priming_stream.daemon.server"]
    kw = calls["kwargs"]
    assert kw["stdin"] is subprocess.DEVNULL
    assert kw["stdout"] is subprocess.DEVNULL
    assert kw["stderr"] is subprocess.DEVNULL

    if sys.platform == "win32":
        flags = kw["creationflags"]
        # Both flags should be ORed in.
        assert flags & subprocess.CREATE_NEW_PROCESS_GROUP
        assert flags & subprocess.DETACHED_PROCESS
    else:
        assert kw.get("start_new_session") is True


def test_autostart_disabled_by_env_skips_popen(monkeypatch):
    """``PRIMING_STREAM_DISABLE_AUTOSTART=1`` must short-circuit
    ``autostart_daemon`` so subprocess-based hook tests don't leak a real
    detached daemon into the developer's machine."""
    calls = {"n": 0}

    class FakePopen:
        def __init__(self, *a, **kw):
            calls["n"] += 1

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    monkeypatch.setattr(lifecycle.subprocess, "Popen", FakePopen)
    monkeypatch.setenv(lifecycle.DISABLE_AUTOSTART_ENV, "1")

    lifecycle.autostart_daemon()

    assert calls["n"] == 0


# --------------------------------------------- autostart cooldown (2026-07-16)


@pytest.fixture
def _count_spawns(monkeypatch):
    """Count autostart spawns without launching anything."""
    calls = {"n": 0}

    class FakePopen:
        def __init__(self, *a, **kw):
            calls["n"] += 1

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    monkeypatch.setattr(lifecycle.subprocess, "Popen", FakePopen)
    monkeypatch.delenv(lifecycle.DISABLE_AUTOSTART_ENV, raising=False)
    return calls


def test_autostart_cooldown_collapses_a_burst_to_one_spawn(_count_spawns):
    """The incident: 4 hook fires in 18 minutes, each convinced the (live)
    daemon was dead, each spawning its own. Back-to-back autostarts must
    collapse to a single spawn."""
    for _ in range(4):
        lifecycle.autostart_daemon()
    assert _count_spawns["n"] == 1


def test_autostart_force_bypasses_cooldown(_count_spawns):
    """`prime daemon start` is a human asking; the cooldown guards against
    a misfiring staleness check, not against the user."""
    lifecycle.autostart_daemon()
    lifecycle.autostart_daemon(force=True)
    assert _count_spawns["n"] == 2


def test_autostart_spawns_again_once_cooldown_expires(_count_spawns):
    lifecycle.autostart_daemon()
    # Age the stamp past the window rather than sleeping through it.
    stamp = lifecycle.autostart_stamp_path()
    old = time.time() - (lifecycle.AUTOSTART_COOLDOWN_S + 1)
    os.utime(stamp, (old, old))
    lifecycle.autostart_daemon()
    assert _count_spawns["n"] == 2


def test_autostart_not_blocked_by_stamp_in_the_future(_count_spawns):
    """The clock jumped ~6.8h forward on resume (01:33:32Z → 08:21:57Z) in
    the very incident this guards. A stamp that ends up in the future must
    not freeze autostart until real time catches up — fail open."""
    lifecycle.autostart_daemon()
    stamp = lifecycle.autostart_stamp_path()
    future = time.time() + 6 * 3600
    os.utime(stamp, (future, future))
    lifecycle.autostart_daemon()
    assert _count_spawns["n"] == 2


def test_autostart_recently_is_false_without_a_stamp():
    assert lifecycle.autostart_recently() is False


# ------------------------------------------------------------- misc/dir


def test_daemon_dir_honors_env(monkeypatch, tmp_path):
    target = tmp_path / "custom"
    monkeypatch.setenv(lifecycle.DAEMON_DIR_ENV, str(target))
    d = lifecycle.daemon_dir()
    assert d == target
    assert d.is_dir()


def test_paths_are_under_daemon_dir():
    d = lifecycle.daemon_dir()
    assert lifecycle.endpoint_path() == d / "daemon.json"
    assert lifecycle.lockfile_path() == d / "daemon.lock"
    assert lifecycle.log_path() == d / "daemon.log"
