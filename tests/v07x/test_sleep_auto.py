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


def test_lock_blocks_concurrent_then_takes_over_stale(tmp_path):
    paths = SimpleNamespace(storage_dir=tmp_path)

    # first acquire succeeds
    assert sleep_auto._acquire_lock(paths, stale_minutes=180) is True
    # a second concurrent acquire is blocked (lock is fresh)
    assert sleep_auto._acquire_lock(paths, stale_minutes=180) is False

    # age the lock past the stale threshold → next acquire takes over
    lock = tmp_path / sleep_auto._LOCK_NAME
    old = time.time() - 999 * 60
    os.utime(lock, (old, old))
    assert sleep_auto._acquire_lock(paths, stale_minutes=180) is True

    sleep_auto._release_lock(paths)
    assert not lock.exists()
    # releasing a non-existent lock is harmless
    sleep_auto._release_lock(paths)
