"""P0 fix batch — unit tests for the 11 robustness fixes.

One module collects all new tests so they are easy to spot. Each test
group is labelled with the fix number it covers.
"""
from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# FIX 1 — bridge/recency.py: divide-by-zero guard in f_recency
# ---------------------------------------------------------------------------

from datetime import datetime, timezone
from priming_stream.bridge.recency import f_recency

_NOW = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_fix1_age_span_zero_returns_one():
    """age_span_days=0 must not raise ZeroDivisionError and must return 1.0."""
    result = f_recency("2026-01-01", _NOW, strength=0.5, age_span_days=0, p_max=0.5)
    assert result == 1.0


def test_fix1_age_span_negative_returns_one():
    result = f_recency("2026-01-01", _NOW, strength=0.5, age_span_days=-10, p_max=0.5)
    assert result == 1.0


# ---------------------------------------------------------------------------
# FIX 2 — core/episodic.py: per-line try/except on corrupt JSON
# ---------------------------------------------------------------------------

from priming_stream.core.episodic import EpisodicStore
from priming_stream.core.models import Chunk, Turn


def _chunk(cid: str) -> Chunk:
    return Chunk(
        chunk_id=cid, source_client="claude_code", session_id="s1",
        started_at="2026-05-20T10:00:00Z", ended_at="2026-05-20T10:05:00Z",
        turns=[Turn(0, "user", "q " + cid, "2026-05-20T10:00:00Z")],
    )


def test_fix2_corrupt_line_skipped_valid_lines_yielded(tmp_path):
    """A corrupt JSON line between two valid lines must not abort iteration;
    both valid lines must be yielded and no exception raised."""
    store = EpisodicStore(tmp_path)
    # Write two valid chunks manually via _append, then corrupt the file.
    store.write_chunk(_chunk("c_before"))
    store.write_chunk(_chunk("c_after"))

    # Inject a corrupt line in the middle by rewriting chunks.jsonl.
    valid1 = (tmp_path / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    corrupt_content = valid1[0] + "\nNOT_JSON_AT_ALL\n" + valid1[1] + "\n"
    (tmp_path / "chunks.jsonl").write_text(corrupt_content, encoding="utf-8")

    store2 = EpisodicStore(tmp_path)
    # Must not raise; must yield both valid chunks.
    chunks = list(store2.iter_chunks())
    assert {c.chunk_id for c in chunks} == {"c_before", "c_after"}


def test_fix2_corrupt_line_emits_stderr_warning(tmp_path, capsys):
    """A corrupt JSON line must emit a warning to stderr."""
    store = EpisodicStore(tmp_path)
    store.write_chunk(_chunk("c_x"))
    lines = (tmp_path / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    corrupt_content = "GARBAGE_JSON\n" + lines[0] + "\n"
    (tmp_path / "chunks.jsonl").write_text(corrupt_content, encoding="utf-8")

    store2 = EpisodicStore(tmp_path)
    list(store2.iter_chunks())
    captured = capsys.readouterr()
    assert "corrupt" in captured.err.lower() or "skip" in captured.err.lower()


# ---------------------------------------------------------------------------
# FIX 3 — core/episodic.py: write_chunk O(N²) cache
# ---------------------------------------------------------------------------

def test_fix3_duplicate_write_is_noop(tmp_path):
    """Two consecutive writes with the same chunk_id: second is no-op."""
    store = EpisodicStore(tmp_path)
    store.write_chunk(_chunk("c_dup"))
    store.write_chunk(_chunk("c_dup"))
    lines = (tmp_path / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_fix3_second_write_does_not_rescan(tmp_path):
    """After the first write populates the cache, subsequent writes must NOT
    call _chunk_ids() (which does a full file scan).

    We monkeypatch _chunk_ids to count calls: after the first write_chunk
    (which triggers lazy-init), a second write_chunk on a NEW id must NOT
    call _chunk_ids() again — the cache is kept in-memory and updated.
    """
    store = EpisodicStore(tmp_path)
    chunk_ids_call_count = 0
    original_chunk_ids = EpisodicStore._chunk_ids

    def counting_chunk_ids(self):
        nonlocal chunk_ids_call_count
        chunk_ids_call_count += 1
        return original_chunk_ids(self)

    with patch.object(EpisodicStore, "_chunk_ids", counting_chunk_ids):
        store2 = EpisodicStore(tmp_path)
        # First write: triggers lazy-init → one _chunk_ids call.
        store2.write_chunk(_chunk("c_first"))
        calls_after_first = chunk_ids_call_count
        # Second write with a DIFFERENT id: cache should be used, no re-scan.
        store2.write_chunk(_chunk("c_second"))
        calls_after_second = chunk_ids_call_count

    # Exactly one full scan (lazy-init on first write), none on subsequent writes.
    assert calls_after_first == 1
    assert calls_after_second == 1  # no additional call


# ---------------------------------------------------------------------------
# FIX 4 — bridge/lexical.py: cursor-level row_factory (no shared-conn mutation)
# ---------------------------------------------------------------------------

import sqlite3
from priming_stream.bridge.lexical import lexical_bucket
from priming_stream.core.db import connect
from priming_stream.core.schema import apply_migrations
from priming_stream.core.models import new_record_id, now_iso


def _seed_db(tmp_path) -> sqlite3.Connection:
    db = tmp_path / "graph.db"
    conn = connect(db)
    apply_migrations(conn)
    rid = new_record_id()
    conn.execute(
        "INSERT INTO records "
        "(id, source_uri, anchor_offset_start, anchor_offset_end, "
        "summary, created_at, source_date, kind, doc_key, source, "
        "content_hash, title, provisional) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (rid, "qmd://test/r.md", 0, 10, "bridge spreading activation test",
         now_iso(), None, "claim", None, None, None, None, 0),
    )
    conn.commit()
    return conn


def test_fix4_lexical_bucket_does_not_mutate_connection_row_factory(tmp_path):
    """lexical_bucket must not change conn.row_factory on the shared connection."""
    conn = _seed_db(tmp_path)
    original_factory = conn.row_factory  # typically None on a fresh connection
    try:
        lexical_bucket(conn, "bridge", limit=5, exclude_ids=set())
        assert conn.row_factory == original_factory
    finally:
        conn.close()


def test_fix4_lexical_bucket_still_returns_results(tmp_path):
    """Sanity: results still come back after the row_factory fix."""
    conn = _seed_db(tmp_path)
    try:
        out = lexical_bucket(conn, "bridge", limit=5, exclude_ids=set())
        assert len(out) == 1
        assert out[0].record.summary == "bridge spreading activation test"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# FIX 5 — cli/sleep_auto.py: shell=False subprocess construction
# ---------------------------------------------------------------------------

from priming_stream.cli import sleep_auto


def test_fix5_shell_false_non_cmd_executable(tmp_path, monkeypatch):
    """With a plain executable (not .cmd/.bat), argv must be a list with
    shell=False — never a string with shell=True."""
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        import subprocess
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(sleep_auto.subprocess, "run", fake_run)
    monkeypatch.setattr(sleep_auto.shutil, "which",
                        lambda cmd: "/usr/bin/claude")

    # Minimal paths/args stubs.
    import argparse
    from types import SimpleNamespace
    args = SimpleNamespace(
        limit=None, claude_cmd="claude", projects_dir=None,
        settled_minutes=30, excludes=[], dry_run=False,
    )
    paths = SimpleNamespace(
        storage_dir=tmp_path,
        graph_db=tmp_path / "graph.db",
        episodic_dir=tmp_path / "episodic",
        corpus_cursor_path=tmp_path / "cursor.json",
    )
    # Patch internal helpers so we reach the subprocess.run call.
    monkeypatch.setattr(sleep_auto, "_acquire_lock", lambda *a, **k: True)
    monkeypatch.setattr(sleep_auto, "_release_lock", lambda *a: None)
    monkeypatch.setattr(sleep_auto, "_settled_sessions", lambda *a, **k: [])
    monkeypatch.setattr(sleep_auto, "_pending_count", lambda *a: 5)

    # Need a real graph.db for the entry check.
    (tmp_path / "graph.db").write_text("", encoding="utf-8")
    (tmp_path / "episodic").mkdir()

    from priming_stream.core.config import load_config
    with patch("priming_stream.cli.sleep_auto.load_config", load_config), \
         patch("priming_stream.cli.sleep_auto.resolve_paths", return_value=paths), \
         patch("priming_stream.cli.sleep_auto.ensure_dirs", return_value=None), \
         patch("priming_stream.cli.sleep_auto.EpisodicStore"):
        sleep_auto.cmd_sleep_auto(args)

    assert "argv" in captured, "subprocess.run was not called"
    assert isinstance(captured["argv"], list), "argv must be a list (shell=False)"
    assert captured["kwargs"].get("shell") is False


def test_fix5_shell_false_cmd_wrapper(tmp_path, monkeypatch):
    """With a .cmd wrapper, argv must start with ['cmd', '/c', ...]."""
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(sleep_auto.subprocess, "run", fake_run)
    monkeypatch.setattr(sleep_auto.shutil, "which",
                        lambda cmd: r"C:\npm\claude.cmd")

    from types import SimpleNamespace
    args = SimpleNamespace(
        limit=None, claude_cmd="claude", projects_dir=None,
        settled_minutes=30, excludes=[], dry_run=False,
    )
    paths = SimpleNamespace(
        storage_dir=tmp_path,
        graph_db=tmp_path / "graph.db",
        episodic_dir=tmp_path / "episodic",
        corpus_cursor_path=tmp_path / "cursor.json",
    )
    monkeypatch.setattr(sleep_auto, "_acquire_lock", lambda *a, **k: True)
    monkeypatch.setattr(sleep_auto, "_release_lock", lambda *a: None)
    monkeypatch.setattr(sleep_auto, "_settled_sessions", lambda *a, **k: [])
    monkeypatch.setattr(sleep_auto, "_pending_count", lambda *a: 5)

    (tmp_path / "graph.db").write_text("", encoding="utf-8")
    (tmp_path / "episodic").mkdir()

    from priming_stream.core.config import load_config
    with patch("priming_stream.cli.sleep_auto.load_config", load_config), \
         patch("priming_stream.cli.sleep_auto.resolve_paths", return_value=paths), \
         patch("priming_stream.cli.sleep_auto.ensure_dirs", return_value=None), \
         patch("priming_stream.cli.sleep_auto.EpisodicStore"):
        sleep_auto.cmd_sleep_auto(args)

    assert "argv" in captured
    assert captured["argv"][:2] == ["cmd", "/c"]
    assert captured["kwargs"].get("shell") is False


def test_fix5_which_returns_none_returns_error(tmp_path, monkeypatch):
    """If shutil.which returns None the function must log and return non-zero."""
    monkeypatch.setattr(sleep_auto.shutil, "which", lambda cmd: None)

    from types import SimpleNamespace
    args = SimpleNamespace(
        limit=None, claude_cmd="claude-nonexistent", projects_dir=None,
        settled_minutes=30, excludes=[], dry_run=False,
    )
    paths = SimpleNamespace(
        storage_dir=tmp_path,
        graph_db=tmp_path / "graph.db",
        episodic_dir=tmp_path / "episodic",
        corpus_cursor_path=tmp_path / "cursor.json",
    )
    monkeypatch.setattr(sleep_auto, "_acquire_lock", lambda *a, **k: True)
    monkeypatch.setattr(sleep_auto, "_release_lock", lambda *a: None)
    monkeypatch.setattr(sleep_auto, "_settled_sessions", lambda *a, **k: [])
    monkeypatch.setattr(sleep_auto, "_pending_count", lambda *a: 5)

    (tmp_path / "graph.db").write_text("", encoding="utf-8")
    (tmp_path / "episodic").mkdir()

    from priming_stream.core.config import load_config
    with patch("priming_stream.cli.sleep_auto.load_config", load_config), \
         patch("priming_stream.cli.sleep_auto.resolve_paths", return_value=paths), \
         patch("priming_stream.cli.sleep_auto.ensure_dirs", return_value=None), \
         patch("priming_stream.cli.sleep_auto.EpisodicStore"):
        rc = sleep_auto.cmd_sleep_auto(args)

    assert rc == 1


# ---------------------------------------------------------------------------
# FIX 7 — _resolve_doc_path: path traversal guard (writer.py in .claude/skills/)
# Applied 2026-06-10 with explicit owner permission (the auto-mode classifier
# blocks unsanctioned edits under .claude/skills/); the script lives in
# prime-ingest/ since the conv-branch consolidation.
# ---------------------------------------------------------------------------

import importlib.util

_WRITER_PATH = (
    Path(__file__).resolve().parents[2]
    / ".claude" / "skills" / "prime-ingest" / "writer.py"
)
_writer_available = _WRITER_PATH.exists()


def _load_writer():
    spec = importlib.util.spec_from_file_location("ch_writer", _WRITER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(
    not _writer_available,
    reason="writer.py not found",
)
def test_fix7_path_traversal_returns_unresolved(tmp_path):
    """LOCALPATH with ../ traversal that escapes cwd must return None."""
    writer = _load_writer()
    cwd = tmp_path / "project"
    cwd.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("sensitive", encoding="utf-8")

    result = writer._resolve_doc_path("..\\secret.txt", str(cwd))
    # With the guard in place: None (traversal blocked).
    # Without the guard: str(secret.resolve()) — the assertion fails.
    assert result is None, (
        "Path traversal was NOT blocked — fix 7 is not yet applied to writer.py"
    )


# ---------------------------------------------------------------------------
# FIX 8 — core/graph_repo.py: LIKE escape
# ---------------------------------------------------------------------------

from priming_stream.core.db import connect as db_connect
from priming_stream.core.graph_repo import GraphRepo
from priming_stream.core.schema import apply_migrations as apply_mig


def _make_repo(tmp_path) -> GraphRepo:
    conn = db_connect(tmp_path / "graph.db")
    apply_mig(conn)
    return GraphRepo(conn)


def test_fix8_underscore_in_prefix_no_wildcard_match(tmp_path):
    """A prefix containing '_' must not wildcard-match unrelated URIs."""
    repo = _make_repo(tmp_path)
    from priming_stream.core.models import Record, new_record_id, now_iso as niso

    def _rec(rid, uri):
        return Record(
            id=rid, source_uri=uri,
            anchor_offset_start=0, anchor_offset_end=1,
            summary="s", created_at=niso(),
        )

    # URI with literal underscore in it.
    repo.create_record(_rec("rec_under01", "conv://sess_1/chunk"))
    # URI that would match if '_' were treated as a wildcard.
    repo.create_record(_rec("rec_wild001", "conv://sessX1/chunk"))
    # URI that matches the literal prefix exactly.
    repo.create_record(_rec("rec_exact01", "conv://sess_1/chunk_two"))

    hits = repo.records_by_source_uri("conv://sess_1/")
    hit_ids = {r.id for r in hits}
    # Must match the exact-prefix records, NOT the wildcard-would-match one.
    assert "rec_wild001" not in hit_ids
    assert "rec_under01" in hit_ids
    assert "rec_exact01" in hit_ids


def test_fix8_percent_in_prefix_literal(tmp_path):
    """A prefix containing '%' must not expand as a SQL wildcard."""
    repo = _make_repo(tmp_path)
    from priming_stream.core.models import Record, new_record_id, now_iso as niso

    repo.create_record(Record(
        id="rec_pct0001", source_uri="file://path%20with%20spaces/doc.pdf",
        anchor_offset_start=0, anchor_offset_end=1,
        summary="s", created_at=niso(),
    ))
    repo.create_record(Record(
        id="rec_nopct001", source_uri="file://pathXXwithXXspaces/doc.pdf",
        anchor_offset_start=0, anchor_offset_end=1,
        summary="s", created_at=niso(),
    ))

    hits = repo.records_by_source_uri("file://path%20with%20spaces/")
    hit_ids = {r.id for r in hits}
    assert "rec_pct0001" in hit_ids
    assert "rec_nopct001" not in hit_ids


# ---------------------------------------------------------------------------
# FIX 9 — hooks/user_prompt_submit.py: abandoned slash commands removed
# (W-G deletion pass: _KNOWN_SLASH_COMMANDS + entire slash-dispatch surface
# have been deleted; these 4 tests are subsumed by that deletion and are
# no longer needed.)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# FIX 10 — core/graph_repo.py: replace_record atomicity
# ---------------------------------------------------------------------------

def test_fix10_replace_record_old_gone_new_present(tmp_path):
    """After replace_record: old id absent, new record present."""
    repo = _make_repo(tmp_path)
    from priming_stream.core.models import Record, new_record_id, now_iso as niso

    old = Record(
        id="rec_old00001", source_uri="qmd://x",
        anchor_offset_start=0, anchor_offset_end=1,
        summary="old summary", created_at=niso(),
        kind="index_card", doc_key="doi:10.1/x", content_hash="hash_old",
    )
    repo.create_record(old)

    new = Record(
        id="rec_new00001", source_uri="qmd://x",
        anchor_offset_start=0, anchor_offset_end=1,
        summary="new summary", created_at=niso(),
        kind="index_card", doc_key="doi:10.1/x", content_hash="hash_new",
    )
    repo.replace_record(old.id, new)

    assert repo.get_record(old.id) is None
    fetched = repo.get_record(new.id)
    assert fetched is not None
    assert fetched.summary == "new summary"
    assert fetched.content_hash == "hash_new"


def test_fix10_replace_record_rollback_on_duplicate_id(tmp_path):
    """If INSERT fails (e.g. new record id already exists), the DELETE must
    be rolled back — the old record must still be present."""
    repo = _make_repo(tmp_path)
    from priming_stream.core.models import Record, now_iso as niso

    # Pre-existing record that will cause the INSERT to fail (duplicate PK).
    existing_blocker = Record(
        id="rec_blocker1", source_uri="qmd://x",
        anchor_offset_start=0, anchor_offset_end=1,
        summary="blocker", created_at=niso(),
    )
    repo.create_record(existing_blocker)

    victim = Record(
        id="rec_victim001", source_uri="qmd://y",
        anchor_offset_start=0, anchor_offset_end=1,
        summary="victim", created_at=niso(),
        kind="index_card", doc_key="doi:10.1/y", content_hash="h1",
    )
    repo.create_record(victim)

    # Try to replace victim with a record whose id collides with existing_blocker.
    new_collision = Record(
        id="rec_blocker1",  # same id as the blocker → INSERT will fail
        source_uri="qmd://y",
        anchor_offset_start=0, anchor_offset_end=1,
        summary="collision", created_at=niso(),
        kind="index_card", doc_key="doi:10.1/y", content_hash="h2",
    )
    with pytest.raises(Exception):
        repo.replace_record(victim.id, new_collision)

    # Victim must still exist (DELETE rolled back).
    assert repo.get_record(victim.id) is not None, (
        "replace_record did not roll back the DELETE on INSERT failure"
    )


# ---------------------------------------------------------------------------
# FIX 11 — daemon/render.py: date-only source_date label
# ---------------------------------------------------------------------------

from priming_stream.daemon.render import render_buckets


def _sem(rid="rec_a", summary="s", source_date=None, kind="claim"):
    return {"record_id": rid, "summary": summary,
            "source_date": source_date, "kind": kind}


def test_fix11_date_only_renders_without_time():
    """A date-only source_date (YYYY-MM-DD, len 10) must render as 'YYYY-MM-DD',
    not 'YYYY-MM-DD 00:00'."""
    out = render_buckets([_sem("rec_a", source_date="2026-01-15")], [])
    assert "[rec_a · 2026-01-15]" in out
    assert "00:00" not in out


def test_fix11_datetime_still_renders_with_time():
    """A full datetime source_date still renders as 'YYYY-MM-DD HH:MM'."""
    out = render_buckets([_sem("rec_b", source_date="2026-01-15T10:30:00Z")], [])
    assert "[rec_b · 2026-01-15 10:30]" in out


def test_fix11_date_only_invalid_falls_back_to_kind_label():
    """An invalid date-only string (e.g. '2026-13-99') falls back to kind label."""
    out = render_buckets([_sem("rec_c", source_date="2026-13-99")], [])
    assert "[rec_c · manual]" in out
