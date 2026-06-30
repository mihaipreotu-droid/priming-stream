"""v0.7-x-vec-index W-A: ``vec-index-rebuild`` CLI + ``prime init`` wiring.

SQL-canonical (2026-06-12): the rebuild source is the SQLite ``records``
table, not ``.md`` files. Uses the stub-embedder pattern: tests monkeypatch
the embedder so the real ~28s fastembed model load is skipped. The contract
under test is the rebuild flow (SELECT + batch) not the embedding itself.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from priming_stream.cli import vec_index as cli_vec_index
from priming_stream.core.db import connect
from priming_stream.core.graph_repo import GraphRepo
from priming_stream.core.models import Record
from priming_stream.core.schema import apply_migrations
from priming_stream.integrations.vec_index import RecordsVecIndex


_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_RUN_VEC = os.environ.get("RUN_VEC_TESTS") == "1"


class _FakeEmbedder:
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
    """Force lazy ``_get_model`` to return the stub instead of fastembed."""
    def fake_get_model(self):
        if self._model is None:
            self._model = _FakeEmbedder(self._model_name)
        return self._model

    monkeypatch.setattr(RecordsVecIndex, "_get_model", fake_get_model)


# -- fixtures ------------------------------------------------------------


def _insert_record(repo: GraphRepo, rec_id: str, summary: str) -> None:
    repo.create_record(Record(
        id=rec_id,
        source_uri="qmd://test/file.md",
        anchor_offset_start=0,
        anchor_offset_end=100,
        summary=summary,
        created_at="2026-05-26T16:00:00Z",
    ))


def _db_with_five(db_path: Path) -> None:
    conn = connect(db_path)
    apply_migrations(conn)
    repo = GraphRepo(conn)
    _insert_record(repo, "rec_001", "alpha decision pilot")
    _insert_record(repo, "rec_002", "beta outcome quality")
    _insert_record(repo, "rec_003", "gamma note architecture")
    _insert_record(repo, "rec_004", "delta sleep cycle skill")
    _insert_record(repo, "rec_005", "epsilon bridge spread")
    conn.close()


# -- C2: rebuild populates 5 entries -------------------------------------


def test_rebuild_populates_five_records(tmp_path: Path):
    db_path = tmp_path / "graph.db"
    persist_dir = tmp_path / "chroma"
    _db_with_five(db_path)

    summary = cli_vec_index.rebuild_vec_index(
        persist_dir=persist_dir,
        db_path=db_path,
        model_name=_MODEL,
    )

    assert summary["rows_scanned"] == 5
    assert summary["records_added"] == 5
    assert summary["empty_skipped"] == 0

    idx = RecordsVecIndex(persist_dir, _MODEL)
    assert idx.count() == 5
    assert idx.has_record("rec_001")
    assert idx.has_record("rec_005")


# -- C3: idempotent re-run -----------------------------------------------


def test_rebuild_idempotent(tmp_path: Path):
    db_path = tmp_path / "graph.db"
    persist_dir = tmp_path / "chroma"
    _db_with_five(db_path)

    cli_vec_index.rebuild_vec_index(
        persist_dir=persist_dir,
        db_path=db_path,
        model_name=_MODEL,
    )
    cli_vec_index.rebuild_vec_index(
        persist_dir=persist_dir,
        db_path=db_path,
        model_name=_MODEL,
    )

    idx = RecordsVecIndex(persist_dir, _MODEL)
    assert idx.count() == 5


def test_rebuild_drops_removed_records(tmp_path: Path):
    """Drop+rebuild semantics: a record deleted from SQLite drops from the
    index on re-run, since the collection is recreated each time."""
    db_path = tmp_path / "graph.db"
    persist_dir = tmp_path / "chroma"
    _db_with_five(db_path)

    cli_vec_index.rebuild_vec_index(
        persist_dir=persist_dir,
        db_path=db_path,
        model_name=_MODEL,
    )

    conn = connect(db_path)
    apply_migrations(conn)
    GraphRepo(conn).delete_record("rec_003")
    conn.close()

    summary = cli_vec_index.rebuild_vec_index(
        persist_dir=persist_dir,
        db_path=db_path,
        model_name=_MODEL,
    )
    assert summary["records_added"] == 4

    idx = RecordsVecIndex(persist_dir, _MODEL)
    assert idx.count() == 4
    assert idx.has_record("rec_003") is False


def test_rebuild_excludes_staged_and_trashed(tmp_path: Path):
    """Only PROMOTED records are embedded — staged (pre-finalize) and
    trashed (soft-deleted) rows never enter the index."""
    db_path = tmp_path / "graph.db"
    persist_dir = tmp_path / "chroma"
    conn = connect(db_path)
    apply_migrations(conn)
    repo = GraphRepo(conn)
    _insert_record(repo, "rec_live", "a live record")
    repo.stage_record(Record(
        id="rec_staged", source_uri="qmd://x/y.md",
        anchor_offset_start=0, anchor_offset_end=1,
        summary="a staged record", created_at="2026-06-01T00:00:00Z",
    ))
    _insert_record(repo, "rec_dead", "a soon-deleted record")
    repo.trash_record("rec_dead", reason="test")
    conn.close()

    summary = cli_vec_index.rebuild_vec_index(
        persist_dir=persist_dir, db_path=db_path, model_name=_MODEL,
    )
    assert summary["records_added"] == 1

    idx = RecordsVecIndex(persist_dir, _MODEL)
    assert idx.has_record("rec_live")
    assert idx.has_record("rec_staged") is False
    assert idx.has_record("rec_dead") is False


# -- C4: empty-summary rows skipped + counted ------------------------------


def test_rebuild_skips_empty_summaries(tmp_path: Path):
    db_path = tmp_path / "graph.db"
    persist_dir = tmp_path / "chroma"
    conn = connect(db_path)
    apply_migrations(conn)
    repo = GraphRepo(conn)
    _insert_record(repo, "rec_good", "valid summary")
    # A whitespace-only summary row (NOT NULL constraint blocks empty at
    # insert; whitespace slips through) — must be skipped, not embedded.
    conn.execute(
        "INSERT INTO records (id, source_uri, summary, created_at) "
        "VALUES ('rec_blank', 'qmd://x/y.md', '   ', '2026-01-01T00:00:00Z')"
    )
    conn.commit()
    conn.close()

    summary = cli_vec_index.rebuild_vec_index(
        persist_dir=persist_dir, db_path=db_path, model_name=_MODEL,
    )

    assert summary["rows_scanned"] == 2
    assert summary["records_added"] == 1
    assert summary["empty_skipped"] == 1

    idx = RecordsVecIndex(persist_dir, _MODEL)
    assert idx.count() == 1
    assert idx.has_record("rec_good")


def test_rebuild_missing_db(tmp_path: Path):
    """A missing database should produce zero records, not crash."""
    persist_dir = tmp_path / "chroma"
    db_path = tmp_path / "does_not_exist.db"

    summary = cli_vec_index.rebuild_vec_index(
        persist_dir=persist_dir, db_path=db_path, model_name=_MODEL,
    )
    assert summary["rows_scanned"] == 0
    assert summary["records_added"] == 0


def test_rebuild_batches_above_50(tmp_path: Path):
    """Cross the BATCH_SIZE=50 boundary to exercise the flush branch."""
    db_path = tmp_path / "graph.db"
    persist_dir = tmp_path / "chroma"
    conn = connect(db_path)
    apply_migrations(conn)
    repo = GraphRepo(conn)
    for i in range(125):
        _insert_record(repo, f"rec_{i:04d}", f"summary number {i}")
    conn.close()

    summary = cli_vec_index.rebuild_vec_index(
        persist_dir=persist_dir, db_path=db_path, model_name=_MODEL,
    )
    assert summary["records_added"] == 125

    idx = RecordsVecIndex(persist_dir, _MODEL)
    assert idx.count() == 125


# -- C1: prime init materializes the dir + empty collection ------------


def test_init_creates_vec_index_dir(tmp_path: Path, monkeypatch):
    """`prime init` in a tmp cwd creates storage/vec_index/chroma."""
    monkeypatch.chdir(tmp_path)

    from priming_stream.cli import main as cli_main

    rc = cli_main.main(["init"])
    assert rc == 0

    vec_dir = tmp_path / "storage" / "vec_index" / "chroma"
    assert vec_dir.is_dir()

    # Opening the index should recover an empty 'records' collection.
    idx = RecordsVecIndex(vec_dir, _MODEL)
    assert idx.count() == 0


# -- real-fastembed end-to-end (slow) ------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not _RUN_VEC, reason="RUN_VEC_TESTS=1 required (model load)")
def test_rebuild_real_fastembed(tmp_path: Path, monkeypatch):
    """Bypass the stub and run with the real embedder; verifies the
    end-to-end CLI flow works against fastembed."""
    monkeypatch.undo()  # release the autouse stub

    db_path = tmp_path / "graph.db"
    persist_dir = tmp_path / "chroma"
    _db_with_five(db_path)

    summary = cli_vec_index.rebuild_vec_index(
        persist_dir=persist_dir, db_path=db_path, model_name=_MODEL,
    )
    assert summary["records_added"] == 5

    idx = RecordsVecIndex(persist_dir, _MODEL)
    assert idx.count() == 5
