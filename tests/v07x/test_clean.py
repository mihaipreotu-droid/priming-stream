"""v0.7-x — ``prime clean-scratch`` removes only regenerable temp artifacts.

Load-bearing safety property: durable state (``_cursor.json``, ``records/``,
``imports/``, snapshots) is NEVER touched; only the explicit scratch allowlist.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from priming_stream.cli import clean
from priming_stream.core.config import load_config
from priming_stream.core.paths import ensure_dirs, resolve_paths


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("PRIMING_STREAM_STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    paths = resolve_paths(cfg)
    ensure_dirs(paths)
    return cfg, paths


def _seed(corpus: Path):
    # scratch (should be removed)
    (corpus / "_sleep_results").mkdir(parents=True, exist_ok=True)
    (corpus / "_sleep_results" / "a.txt").write_text("x", encoding="utf-8")
    (corpus / "_doc_md").mkdir(parents=True, exist_ok=True)
    (corpus / "_produced_docs.json").write_text("[]", encoding="utf-8")
    (corpus / "_reconcile_plan.json").write_text("{}", encoding="utf-8")
    (corpus / "_sleep_manifest.json").write_text("{}", encoding="utf-8")
    # durable (must survive)
    (corpus / "_cursor.json").write_text('{"last_chunk_id":"c1"}', encoding="utf-8")
    (corpus / "records").mkdir(parents=True, exist_ok=True)
    (corpus / "records" / "rec_x.md").write_text("---\nid: rec_x\n---\nbody", encoding="utf-8")
    (corpus / "imports").mkdir(parents=True, exist_ok=True)
    (corpus / "imports" / "chunk.md").write_text("c", encoding="utf-8")
    (corpus / "_pre_W7_snapshot").mkdir(parents=True, exist_ok=True)


def test_clean_scratch_removes_temp_keeps_durable(env):
    _cfg, paths = env
    corpus = paths.corpus_dir
    _seed(corpus)

    rc = clean.cmd_clean_scratch(argparse.Namespace(dry_run=False))
    assert rc == 0

    # scratch gone
    for name in ("_sleep_results", "_doc_md", "_produced_docs.json",
                 "_reconcile_plan.json", "_sleep_manifest.json"):
        assert not (corpus / name).exists(), name
    # durable untouched
    assert (corpus / "_cursor.json").exists()
    assert (corpus / "records" / "rec_x.md").exists()
    assert (corpus / "imports" / "chunk.md").exists()
    assert (corpus / "_pre_W7_snapshot").exists()


def test_clean_scratch_dry_run_deletes_nothing(env):
    _cfg, paths = env
    corpus = paths.corpus_dir
    _seed(corpus)

    rc = clean.cmd_clean_scratch(argparse.Namespace(dry_run=True))
    assert rc == 0
    # everything still there after a dry-run
    assert (corpus / "_sleep_results").exists()
    assert (corpus / "_produced_docs.json").exists()


# -- clean-cc-subagents — deletion guards (CRITICAL) ---------------------


import os
import time


def _touch_old(p: Path, days_old: float):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{}\n", encoding="utf-8")
    t = time.time() - days_old * 86400
    os.utime(p, (t, t))


def test_clean_cc_subagents_double_guard(tmp_path):
    proj = tmp_path / "projects"
    # a MAIN session transcript (old) — MUST NEVER be deleted
    main = proj / "projA" / "11111111-2222-3333-4444-555555555555.jsonl"
    _touch_old(main, days_old=10)
    # an OLD sub-agent transcript — the only legitimate victim
    old_sub = proj / "projA" / "sess" / "subagents" / "agent-old.jsonl"
    _touch_old(old_sub, days_old=10)
    # a RECENT sub-agent — kept (under the threshold)
    new_sub = proj / "projA" / "sess" / "subagents" / "agent-new.jsonl"
    _touch_old(new_sub, days_old=0.1)
    # a non-agent file under subagents (old) — kept (name guard)
    other = proj / "projA" / "sess" / "subagents" / "notes.jsonl"
    _touch_old(other, days_old=10)

    rc = clean.cmd_clean_cc_subagents(argparse.Namespace(
        older_than=2, projects_dir=str(proj), execute=True))
    assert rc == 0

    assert not old_sub.exists(), "old sub-agent should be pruned"
    assert main.exists(), "MAIN session transcript must survive"
    assert new_sub.exists(), "recent sub-agent must survive"
    assert other.exists(), "non-agent-* file must survive"


def test_clean_cc_subagents_dry_run_deletes_nothing(tmp_path):
    proj = tmp_path / "projects"
    old_sub = proj / "p" / "s" / "subagents" / "agent-x.jsonl"
    _touch_old(old_sub, days_old=10)

    rc = clean.cmd_clean_cc_subagents(argparse.Namespace(
        older_than=2, projects_dir=str(proj), execute=False))
    assert rc == 0
    assert old_sub.exists(), "dry-run must not delete"
