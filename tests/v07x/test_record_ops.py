"""v0.7-x record curation CLI — ``prime record create|edit|delete|restore``.

Covers the owner-curation surface (`cli/record_ops.py`), SQL-canonical:

- create writes a new owner-authored claim into SQLite (source of truth) +
  ChromaDB, anchored ``owner://`` 0/0 and stamped ``source_date = created_at``.
- edit rewrites SQLite (FTS5 synced by the ``records_au`` trigger) + ChromaDB
  (re-embed).
- delete: soft moves the row to the ``records_trash`` table (reversible via
  ``record restore``); ``--hard`` deletes the row outright. Either way the
  embedding is dropped.
- the load-bearing correctness properties: an edit and a soft-delete both
  SURVIVE ``vec-index-rebuild`` (which rebuilds Chroma from SQLite).
- guards: ``index_card`` edit refused, missing record, empty summary.

Stub-embedder pattern (mirrors ``test_vec_index_rebuild``): the real
ChromaDB runs, only the ~28s fastembed model is replaced by a
deterministic hash embedder. The daemon reload is stubbed so the tests
never touch a live daemon. Storage is isolated via ``PRIMING_STREAM_STORAGE_DIR``.
"""
from __future__ import annotations

import argparse
import hashlib

import pytest

from priming_stream.cli import record_ops
from priming_stream.cli import vec_index as cli_vec_index
from priming_stream.core.config import load_config
from priming_stream.core.db import connect
from priming_stream.core.graph_repo import GraphRepo
from priming_stream.core.models import Record, now_iso
from priming_stream.core.paths import ensure_dirs, resolve_paths
from priming_stream.core.schema import apply_migrations
from priming_stream.integrations.vec_index import RecordsVecIndex


# -- stubs ---------------------------------------------------------------


class _FakeEmbedder:
    """Deterministic 32-dim embedder — real ChromaDB, no fastembed load."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def embed(self, texts):
        import numpy as np

        for t in texts:
            digest = hashlib.sha256(t.encode("utf-8")).digest()
            arr = np.frombuffer(digest, dtype=np.uint8).astype("float32")
            norm = np.linalg.norm(arr) or 1.0
            yield arr / norm


@pytest.fixture(autouse=True)
def _stub_embedder(monkeypatch):
    def fake_get_model(self):
        if self._model is None:
            self._model = _FakeEmbedder(self._model_name)
        return self._model

    monkeypatch.setattr(RecordsVecIndex, "_get_model", fake_get_model)


@pytest.fixture(autouse=True)
def _no_daemon(monkeypatch):
    """record_ops._reload_daemon does a deferred import of the daemon
    client; patch the function it calls so tests never hit a live daemon."""
    monkeypatch.setattr(
        "priming_stream.daemon.client.reload_daemon", lambda timeout_s=5.0: None
    )


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated storage rooted at tmp_path; returns (cfg, paths)."""
    monkeypatch.setenv("PRIMING_STREAM_STORAGE_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    paths = resolve_paths(cfg)
    ensure_dirs(paths)
    conn = connect(paths.graph_db)
    apply_migrations(conn)
    conn.close()
    return cfg, paths


# -- helpers -------------------------------------------------------------


def _make_record(cfg, paths, *, rid="rec_aaaa0001", summary="alpha original summary",
                 kind="claim", doc_key=None):
    """Create a record across SQLite + ChromaDB, like the create path."""
    rec = Record(
        id=rid, source_uri="test://x", anchor_offset_start=None,
        anchor_offset_end=None, summary=summary, created_at=now_iso(),
        kind=kind, doc_key=doc_key,
    )
    conn = connect(paths.graph_db)
    apply_migrations(conn)
    GraphRepo(conn).create_record(rec)
    conn.close()
    RecordsVecIndex(paths.vec_index_dir, cfg.vec_index.model_name).add_record(rid, summary)
    return rec


def _get_record(paths, rid):
    conn = connect(paths.graph_db)
    apply_migrations(conn)
    try:
        return GraphRepo(conn).get_record(rid)
    finally:
        conn.close()


def _summary_of(paths, rid):
    rec = _get_record(paths, rid)
    return rec.summary if rec else None


def _trashed(paths, rid):
    conn = connect(paths.graph_db)
    apply_migrations(conn)
    try:
        return GraphRepo(conn).get_trashed(rid)
    finally:
        conn.close()


def _chroma_doc(cfg, paths, rid):
    got = RecordsVecIndex(paths.vec_index_dir, cfg.vec_index.model_name)._collection.get(ids=[rid])
    docs = got.get("documents") or []
    return docs[0] if docs else None


def _only_record(paths) -> Record:
    """Return the single record row in the substrate."""
    conn = connect(paths.graph_db)
    apply_migrations(conn)
    try:
        recs = GraphRepo(conn).list_records()
    finally:
        conn.close()
    assert len(recs) == 1, f"expected one record, found {len(recs)}"
    return recs[0]


# -- create --------------------------------------------------------------


def test_create_writes_both_stores(env):
    cfg, paths = env

    rc = record_ops._cmd_create(argparse.Namespace(
        text="DECIS: owner-authored ground truth", summary_file=None))
    assert rc == 0

    rec = _only_record(paths)
    # SQLite (source of truth)
    assert rec.summary == "DECIS: owner-authored ground truth"
    # ChromaDB
    assert _chroma_doc(cfg, paths, rec.id) == "DECIS: owner-authored ground truth"


def test_create_stamps_source_date_and_owner_anchor(env):
    cfg, paths = env

    record_ops._cmd_create(argparse.Namespace(
        text="alpha owner claim", summary_file=None))
    rec = _only_record(paths)

    # owner-authored anchor + recency stamp (roadmap W.1)
    assert rec.source_uri == "owner://"
    assert rec.kind == "claim"
    assert rec.anchor_offset_start == 0
    assert rec.anchor_offset_end == 0
    assert rec.source_date is not None
    assert rec.source_date == rec.created_at  # same instant


def test_create_is_live_immediately_no_staging(env):
    """Owner curation bypasses staging — the record is live (searchable)
    without waiting for a sleep cycle."""
    cfg, paths = env
    record_ops._cmd_create(argparse.Namespace(
        text="delta immediate claim", summary_file=None))
    rec = _only_record(paths)

    conn = connect(paths.graph_db)
    apply_migrations(conn)
    try:
        repo = GraphRepo(conn)
        assert repo.get_staged(rec.id) is None
        hits = conn.execute(
            "SELECT 1 FROM records_fts WHERE records_fts MATCH 'delta'"
        ).fetchall()
    finally:
        conn.close()
    assert len(hits) == 1  # FTS5 live via the records_ai trigger


def test_create_survives_vec_rebuild(env):
    cfg, paths = env

    record_ops._cmd_create(argparse.Namespace(
        text="charlie rebuilt claim", summary_file=None))
    rec = _only_record(paths)

    cli_vec_index.rebuild_vec_index(
        persist_dir=paths.vec_index_dir,
        db_path=paths.graph_db,
        model_name=cfg.vec_index.model_name,
    )
    assert _chroma_doc(cfg, paths, rec.id) == "charlie rebuilt claim"


def test_create_from_summary_file(env, tmp_path):
    cfg, paths = env
    f = tmp_path / "note.txt"
    payload = 'from file — em–dash, "quotes", and a $var'
    f.write_text(payload, encoding="utf-8")

    rc = record_ops._cmd_create(argparse.Namespace(
        text=None, summary_file=str(f)))
    assert rc == 0
    rec = _only_record(paths)
    assert rec.summary == payload


def test_create_empty_returns_1(env):
    cfg, paths = env
    rc = record_ops._cmd_create(argparse.Namespace(text="   ", summary_file=None))
    assert rc == 1
    conn = connect(paths.graph_db)
    apply_migrations(conn)
    try:
        assert GraphRepo(conn).list_records() == []
    finally:
        conn.close()


# -- edit ----------------------------------------------------------------


def test_edit_updates_both_stores(env):
    cfg, paths = env
    rec = _make_record(cfg, paths)

    rc = record_ops._cmd_edit(argparse.Namespace(
        record_id=rec.id, summary="bravo edited summary", summary_file=None))
    assert rc == 0

    edited = _get_record(paths, rec.id)
    assert edited.summary == "bravo edited summary"
    # identity fields preserved
    assert edited.source_uri == "test://x"
    assert edited.created_at == rec.created_at

    assert _chroma_doc(cfg, paths, rec.id) == "bravo edited summary"


def test_edit_syncs_fts5(env):
    cfg, paths = env
    rec = _make_record(cfg, paths, summary="zappa original lexeme")

    record_ops._cmd_edit(argparse.Namespace(
        record_id=rec.id, summary="quux edited lexeme", summary_file=None))

    conn = connect(paths.graph_db)
    apply_migrations(conn)
    try:
        new = conn.execute(
            "SELECT 1 FROM records_fts WHERE records_fts MATCH ?", ("quux",)).fetchall()
        old = conn.execute(
            "SELECT 1 FROM records_fts WHERE records_fts MATCH ?", ("zappa",)).fetchall()
    finally:
        conn.close()
    assert len(new) == 1
    assert old == []


def test_edit_from_summary_file(env, tmp_path):
    cfg, paths = env
    rec = _make_record(cfg, paths)
    f = tmp_path / "new_summary.txt"
    payload = 'from file — em–dash, "quotes", and a $var'
    f.write_text(payload, encoding="utf-8")

    rc = record_ops._cmd_edit(argparse.Namespace(
        record_id=rec.id, summary=None, summary_file=str(f)))
    assert rc == 0
    assert _summary_of(paths, rec.id) == payload


def test_edit_refuses_index_card(env):
    cfg, paths = env
    rec = _make_record(
        cfg, paths, rid="rec_card0001", summary="card summary",
        kind="index_card", doc_key="dk1")

    rc = record_ops._cmd_edit(argparse.Namespace(
        record_id=rec.id, summary="should not apply", summary_file=None))
    assert rc == 1
    assert _summary_of(paths, rec.id) == "card summary"  # untouched


def test_edit_missing_record_returns_1(env):
    cfg, paths = env
    rc = record_ops._cmd_edit(argparse.Namespace(
        record_id="rec_missing0", summary="x", summary_file=None))
    assert rc == 1


def test_edit_empty_summary_returns_1(env):
    cfg, paths = env
    rec = _make_record(cfg, paths)
    rc = record_ops._cmd_edit(argparse.Namespace(
        record_id=rec.id, summary="   ", summary_file=None))
    assert rc == 1
    assert _summary_of(paths, rec.id) == "alpha original summary"


def test_edit_survives_rebuild(env):
    """Load-bearing: rebuild reconstructs Chroma from SQLite. The edit lands
    in SQLite first, so the rebuilt embedding reflects it."""
    cfg, paths = env
    rec = _make_record(cfg, paths)

    record_ops._cmd_edit(argparse.Namespace(
        record_id=rec.id, summary="bravo edited summary", summary_file=None))
    cli_vec_index.rebuild_vec_index(
        persist_dir=paths.vec_index_dir,
        db_path=paths.graph_db,
        model_name=cfg.vec_index.model_name,
    )
    assert _chroma_doc(cfg, paths, rec.id) == "bravo edited summary"


# -- delete / restore -------------------------------------------------------


def test_soft_delete_moves_row_to_trash(env):
    cfg, paths = env
    rec = _make_record(cfg, paths)

    rc = record_ops._cmd_delete(argparse.Namespace(record_id=rec.id, hard=False))
    assert rc == 0

    assert _summary_of(paths, rec.id) is None
    trashed = _trashed(paths, rec.id)
    assert trashed is not None
    assert trashed.summary == "alpha original summary"
    assert not RecordsVecIndex(
        paths.vec_index_dir, cfg.vec_index.model_name).has_record(rec.id)


def test_soft_delete_survives_rebuild(env):
    """Load-bearing: a trashed row is outside ``records``, so the rebuild
    (SELECT FROM records) must not resurrect it."""
    cfg, paths = env
    rec = _make_record(cfg, paths)

    record_ops._cmd_delete(argparse.Namespace(record_id=rec.id, hard=False))
    cli_vec_index.rebuild_vec_index(
        persist_dir=paths.vec_index_dir,
        db_path=paths.graph_db,
        model_name=cfg.vec_index.model_name,
    )
    assert not RecordsVecIndex(
        paths.vec_index_dir, cfg.vec_index.model_name).has_record(rec.id)


def test_hard_delete_skips_trash(env):
    cfg, paths = env
    rec = _make_record(cfg, paths)

    rc = record_ops._cmd_delete(argparse.Namespace(record_id=rec.id, hard=True))
    assert rc == 0
    assert _summary_of(paths, rec.id) is None
    assert _trashed(paths, rec.id) is None  # hard = no trash row


def test_delete_missing_record_returns_1(env):
    cfg, paths = env
    rc = record_ops._cmd_delete(argparse.Namespace(record_id="rec_missing0", hard=False))
    assert rc == 1


def test_restore_reverses_soft_delete(env):
    cfg, paths = env
    rec = _make_record(cfg, paths)
    record_ops._cmd_delete(argparse.Namespace(record_id=rec.id, hard=False))
    assert _summary_of(paths, rec.id) is None

    rc = record_ops._cmd_restore(argparse.Namespace(record_id=rec.id))
    assert rc == 0
    restored = _get_record(paths, rec.id)
    assert restored is not None
    assert restored.summary == "alpha original summary"
    assert _trashed(paths, rec.id) is None
    # re-embedded
    assert _chroma_doc(cfg, paths, rec.id) == "alpha original summary"


def test_restore_missing_returns_1(env):
    cfg, paths = env
    rc = record_ops._cmd_restore(argparse.Namespace(record_id="rec_nothere0"))
    assert rc == 1
