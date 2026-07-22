"""v0.7-x W-automation — ``prime sleep-auto`` deterministic logic.

Covers the parts that run WITHOUT the LLM: session discovery (settled-only,
sub-agent transcripts excluded, substring excludes) and the concurrency lock.
The LLM step (``claude -p "/prime-ingest"``) is not exercised here.
"""
from __future__ import annotations

import os
import time
from types import SimpleNamespace

from priming_stream.cli import sleep_auto


# -- discovery -----------------------------------------------------------


def _touch(p, age_seconds: float) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{}\n", encoding="utf-8")
    t = time.time() - age_seconds
    os.utime(p, (t, t))


def test_settled_sessions_filters_correctly(tmp_path):
    proj = tmp_path / "projects"
    # a settled main session → INCLUDED
    main_old = proj / "projA" / "sess-old.jsonl"
    _touch(main_old, 3600)  # 1h old
    # a sub-agent transcript (under subagents/) → EXCLUDED
    sub = proj / "projA" / "sess-old" / "subagents" / "agent-x.jsonl"
    _touch(sub, 3600)
    # a fresh (in-progress) session → EXCLUDED (not settled)
    fresh = proj / "projB" / "sess-fresh.jsonl"
    _touch(fresh, 60)  # 1 min old
    # an excluded-by-substring session → EXCLUDED
    excl = proj / "ch_auto" / "sess-auto.jsonl"
    _touch(excl, 3600)

    got = sleep_auto._settled_sessions(proj, settled_minutes=30, excludes=["ch_auto"])
    names = {p.name for p in got}

    assert names == {"sess-old.jsonl"}, names
    assert not any("subagents" in p.parts for p in got)


def test_settled_sessions_missing_dir_is_empty(tmp_path):
    assert sleep_auto._settled_sessions(
        tmp_path / "nope", settled_minutes=30, excludes=[]
    ) == []


# -- lock ----------------------------------------------------------------


def test_lock_is_real_os_lock(tmp_path):
    """The write-cycle lock is a REAL OS lock — atomic
    acquire, no TOCTOU, released by release (or by process death). The
    lockFILE persists between cycles; existence means nothing."""
    from priming_stream.core import write_lock

    paths = SimpleNamespace(storage_dir=tmp_path)
    lock = tmp_path / sleep_auto._LOCK_NAME

    # first acquire succeeds and is observable via the probe
    assert sleep_auto._acquire_lock(paths) is True
    assert write_lock.is_held(tmp_path) is True
    # a concurrent acquire (fresh handle, same byte-0 region) is blocked
    assert write_lock.acquire(tmp_path) is None

    sleep_auto._release_lock(paths)
    # the OS lock is gone; the FILE deliberately persists (probe, not exists)
    assert write_lock.is_held(tmp_path) is False
    assert lock.exists()
    # re-acquire after release works; releasing twice is harmless
    assert sleep_auto._acquire_lock(paths) is True
    sleep_auto._release_lock(paths)
    sleep_auto._release_lock(paths)
