"""v0.7-x-vec-index W-C: ``prime sleep-prepare`` + ``prime sleep-finalize`` CLI.

sleep-prepare in v0.7-x-vec-index has no external indexer step (qmd is
gone, the vec_index only carries records and sleep-prepare only handles
chunks). It materializes chunks to .md, opens a sleep_cycles row, and
prints + persists the JSON manifest; the cursor commit is DEFERRED to
sleep-finalize (crash-safety, 2026-06-10) which reads the manifest back.

sleep-finalize reconciles records into SQLite then pushes them into the
ChromaDB ``records`` collection via :class:`RecordsVecIndex`. The
``--skip-vec-index`` flag is preserved for tests that don't want to load
fastembed.
"""
from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pytest

from priming_stream.cli import sleep as sleep_cli
from priming_stream.daemon import lifecycle
from priming_stream.core.db import connect
from priming_stream.core.episodic import EpisodicStore
from priming_stream.core.graph_repo import GraphRepo
from priming_stream.core.models import Chunk, Turn
from priming_stream.core.schema import apply_migrations


# -- helpers --------------------------------------------------------------


def _chunk(chunk_id: str, n_turns: int = 2) -> Chunk:
    turns = [
        Turn(
            index=i,
            role="user" if i % 2 == 0 else "assistant",
            text=f"turn-{i} body",
            timestamp=f"2026-05-25T10:00:{i:02d}Z",
        )
        for i in range(n_turns)
    ]
    return Chunk(
        chunk_id=chunk_id,
        source_client="claude_ai_export",
        session_id="sess-aaaa",
        started_at="2026-05-25T10:00:00Z",
        ended_at="2026-05-25T10:01:00Z",
        turns=turns,
    )


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Create a project root with initialized storage; chdir into it.

    Mirrors how the CLI is actually run (cwd = project root). All paths
    in the config default to ``storage/...`` relative to cwd.
    """
    project_root = tmp_path / "proj"
    project_root.mkdir()
    monkeypatch.chdir(project_root)

    # Isolate the daemon discovery dir so cmd_sleep_finalize's reload call
    # finds no endpoint and silently skips — otherwise these tests hit a
    # real daemon (if one is running on the dev box), which appends a
    # "[sleep-finalize] daemon reloaded" line after the JSON and breaks
    # json.loads on stdout. The reload integration has its own tests
    # (test_sleep_finalize_reloads_daemon.py, R5); here it must be inert.
    monkeypatch.setenv(lifecycle.DAEMON_DIR_ENV, str(tmp_path / "daemon_iso"))

    # Initialize storage + DB.
    storage = project_root / "storage"
    storage.mkdir()
    db = storage / "graph.db"
    conn = connect(db)
    try:
        apply_migrations(conn)
    finally:
        conn.close()

    return project_root


def _seed_chunks(project_root: Path, ids: list[str]) -> None:
    episodic = EpisodicStore(project_root / "storage" / "episodic")
    for cid in ids:
        episodic.write_chunk(_chunk(cid))


def _prep_args(**kw) -> argparse.Namespace:
    base = dict(limit=None, all_pending=True, no_materialize=False)
    base.update(kw)
    return argparse.Namespace(**base)


def _finalize_args(cycle_id: int, **overrides) -> argparse.Namespace:
    base = dict(
        cycle_id=cycle_id,
        chunks_materialized=0,
        records_created=0,
        records_skipped=0,
        notes=None,
        skip_vec_index=True,
        manifest_path=None,  # no manifest by default (cursor-commit skipped)
    )
    base.update(overrides)
    return argparse.Namespace(**base)


# -- sleep-prepare --------------------------------------------------------


def test_sleep_prepare_emits_manifest(project):
    _seed_chunks(project, ["export_a_p0", "export_a_p1", "export_a_p2"])
    args = _prep_args(limit=5, all_pending=False)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = sleep_cli.cmd_sleep_prepare(args)
    assert rc == 0

    manifest = json.loads(buf.getvalue())
    assert "cycle_id" in manifest
    assert isinstance(manifest["cycle_id"], int)
    assert manifest["in_place_docs"] == []
    assert len(manifest["prepared_chunks"]) == 3

    prepared = manifest["prepared_chunks"]
    chunk_ids = {c["chunk_id"] for c in prepared}
    assert chunk_ids == {"export_a_p0", "export_a_p1", "export_a_p2"}
    for c in prepared:
        assert Path(c["path"]).exists()
        # source_uri still uses the qmd:// scheme name (historical / stable).
        assert c["source_uri"].startswith(
            "qmd://priming-stream-imports/"
        )


def test_sleep_prepare_respects_limit(project):
    _seed_chunks(project, [f"export_a_p{i}" for i in range(5)])
    args = _prep_args(limit=2, all_pending=False)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = sleep_cli.cmd_sleep_prepare(args)
    assert rc == 0
    manifest = json.loads(buf.getvalue())
    assert len(manifest["prepared_chunks"]) == 2


def test_sleep_prepare_opens_cycle_row(project):
    _seed_chunks(project, ["export_a_p0"])
    args = _prep_args(limit=5, all_pending=False)
    buf = io.StringIO()
    with redirect_stdout(buf):
        sleep_cli.cmd_sleep_prepare(args)
    manifest = json.loads(buf.getvalue())

    conn = connect(project / "storage" / "graph.db")
    try:
        repo = GraphRepo(conn)
        cycles = repo.list_sleep_cycles()
    finally:
        conn.close()
    row = next(c for c in cycles if c["id"] == manifest["cycle_id"])
    assert row["started_at"]
    assert row["completed_at"] is None  # still open


def test_sleep_prepare_no_pending_emits_empty_manifest(project):
    args = _prep_args()
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = sleep_cli.cmd_sleep_prepare(args)
    assert rc == 0
    manifest = json.loads(buf.getvalue())
    assert manifest["prepared_chunks"] == []
    # Cycle row is opened regardless (skill still reports / finalizes).
    assert isinstance(manifest["cycle_id"], int)


def test_sleep_prepare_cursor_advances(project):
    """Cursor does NOT advance during prepare (crash-safety); it advances
    during finalize when the manifest is present."""
    _seed_chunks(project, ["export_a_p0", "export_a_p1"])
    cursor_path = project / "storage" / "corpus" / "_cursor.json"
    manifest_path = project / "storage" / "corpus" / "_sleep_manifest.json"

    # --- prepare: cursor must NOT move ---
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = sleep_cli.cmd_sleep_prepare(_prep_args())
    assert rc == 0
    manifest = json.loads(buf.getvalue())
    assert len(manifest["prepared_chunks"]) == 2
    # Cursor still absent — prepare deliberately does not commit it.
    assert not cursor_path.exists()

    # Persist manifest to the standard location so finalize can read it.
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    # --- finalize: cursor MUST advance ---
    fin_args = _finalize_args(manifest["cycle_id"], manifest_path=str(manifest_path))
    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        rc2 = sleep_cli.cmd_sleep_finalize(fin_args)
    assert rc2 == 0
    assert cursor_path.exists()
    state = json.loads(cursor_path.read_text(encoding="utf-8"))
    assert state["last_chunk_id"] == "export_a_p1"


def test_sleep_prepare_no_materialize_opens_empty_cycle_without_cursor(project):
    """piece3: --no-materialize opens a cycle but does NOT drain pending
    chunks or advance the cursor (so /prime-ingest can reconcile cards without
    silently skipping conversational extraction of pending chunks)."""
    _seed_chunks(project, ["export_a_p0", "export_a_p1"])
    cursor_path = project / "storage" / "corpus" / "_cursor.json"

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = sleep_cli.cmd_sleep_prepare(_prep_args(no_materialize=True))
    assert rc == 0
    manifest = json.loads(buf.getvalue())
    # Empty cycle: nothing materialized.
    assert manifest["prepared_chunks"] == []
    assert isinstance(manifest["cycle_id"], int)
    # Cursor untouched — the 2 chunks are still pending for a real cycle.
    assert not cursor_path.exists()

    # A subsequent REAL prepare still sees both chunks as pending.
    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        sleep_cli.cmd_sleep_prepare(_prep_args())
    assert len(json.loads(buf2.getvalue())["prepared_chunks"]) == 2


def test_sleep_prepare_idempotent_second_run(project):
    """Regression (spec P1-B §d): prepare without a subsequent finalize leaves
    the cursor unmoved, so a second prepare re-materialises the SAME chunks
    (idempotent overwrite on disk) — nothing is lost on crash."""
    _seed_chunks(project, ["export_a_p0", "export_a_p1"])
    args = _prep_args()
    buf1 = io.StringIO()
    with redirect_stdout(buf1):
        rc1 = sleep_cli.cmd_sleep_prepare(args)
    assert rc1 == 0
    m1 = json.loads(buf1.getvalue())
    assert len(m1["prepared_chunks"]) == 2

    # No finalize between the two prepares — cursor still at zero.
    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        rc2 = sleep_cli.cmd_sleep_prepare(args)
    assert rc2 == 0
    m2 = json.loads(buf2.getvalue())
    # Same two chunks must be returned (not an empty list).
    assert len(m2["prepared_chunks"]) == 2
    assert {c["chunk_id"] for c in m2["prepared_chunks"]} == {
        c["chunk_id"] for c in m1["prepared_chunks"]
    }


# -- sleep-finalize -------------------------------------------------------


def test_sleep_finalize_closes_row(project):
    # Open a row via the repo directly (sleep-prepare's only side effect
    # we care about for this test is the row).
    conn = connect(project / "storage" / "graph.db")
    try:
        repo = GraphRepo(conn)
        cycle_id = repo.start_sleep_cycle(started_at="2026-05-25T10:00:00Z")
    finally:
        conn.close()

    args = _finalize_args(
        cycle_id,
        chunks_materialized=3,
        records_created=4,
        records_skipped=1,
        notes="smoke",
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = sleep_cli.cmd_sleep_finalize(args)
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["cycle_id"] == cycle_id
    assert payload["completed_at"]

    conn = connect(project / "storage" / "graph.db")
    try:
        repo = GraphRepo(conn)
        cycles = repo.list_sleep_cycles()
    finally:
        conn.close()
    row = next(c for c in cycles if c["id"] == cycle_id)
    assert row["completed_at"]
    assert row["chunks_materialized"] == 3
    assert row["records_created"] == 4
    assert row["records_skipped"] == 1
    assert row["notes"] == "smoke"
    assert row["metrics_json"]
    metrics = json.loads(row["metrics_json"])
    assert metrics["chunks_materialized"] == 3
    assert metrics["records_created"] == 4
    assert metrics["records_skipped"] == 1
    assert "elapsed_s" in metrics


def test_sleep_finalize_unknown_cycle_id(project):
    args = _finalize_args(9999)
    rc = sleep_cli.cmd_sleep_finalize(args)
    assert rc == 1


def test_sleep_finalize_already_completed(project):
    conn = connect(project / "storage" / "graph.db")
    try:
        repo = GraphRepo(conn)
        cycle_id = repo.start_sleep_cycle(started_at="2026-05-25T10:00:00Z")
        repo.finish_sleep_cycle(
            cycle_id,
            completed_at="2026-05-25T10:05:00Z",
            chunks_materialized=0,
            records_created=0,
            records_skipped=0,
            metrics_json="{}",
            notes=None,
        )
    finally:
        conn.close()

    args = _finalize_args(cycle_id)
    rc = sleep_cli.cmd_sleep_finalize(args)
    assert rc == 1


# -- F-1: staged-records promotion in sleep-finalize -----------------------


def _stage(
    project_root: Path, rid: str, *,
    source_uri: str = "qmd://priming-stream-imports/x/y/z.md",
    anchor_start: int | None = 100,
    anchor_end: int | None = 200,
    created_at: str = "2026-05-25T10:00:00Z",
    summary: str = "default summary",
    **extra,
) -> None:
    """Stage one record row (the bulk-writer's output shape)."""
    from priming_stream.core.models import Record

    conn = connect(project_root / "storage" / "graph.db")
    try:
        GraphRepo(conn).stage_record(Record(
            id=rid, source_uri=source_uri,
            anchor_offset_start=anchor_start, anchor_offset_end=anchor_end,
            summary=summary, created_at=created_at, **extra,
        ))
    finally:
        conn.close()


def _open_cycle(project_root: Path) -> int:
    conn = connect(project_root / "storage" / "graph.db")
    try:
        repo = GraphRepo(conn)
        return repo.start_sleep_cycle(started_at="2026-05-25T10:00:00Z")
    finally:
        conn.close()


def test_sleep_finalize_promotes_staged_records(project):
    """F-1: staged rows get promoted into the canonical records table."""
    _stage(project, "rec_aaaaaaa1", summary="alpha summary")
    _stage(project, "rec_bbbbbbb2", summary="beta summary")
    _stage(project, "rec_ccccccc3", summary="gamma summary")

    cycle_id = _open_cycle(project)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = sleep_cli.cmd_sleep_finalize(_finalize_args(cycle_id))
    assert rc == 0

    conn = connect(project / "storage" / "graph.db")
    try:
        repo = GraphRepo(conn)
        ids = {r.id for r in repo.list_records()}
        rec = repo.get_record("rec_aaaaaaa1")
        staged_left = repo.list_staged()
    finally:
        conn.close()
    assert ids == {"rec_aaaaaaa1", "rec_bbbbbbb2", "rec_ccccccc3"}

    # The promoted record's fields carried over intact.
    assert rec is not None
    assert rec.summary == "alpha summary"
    assert rec.source_uri == "qmd://priming-stream-imports/x/y/z.md"
    assert rec.anchor_offset_start == 100
    assert rec.anchor_offset_end == 200

    # Staging drained; metrics record the promotion count.
    assert staged_left == []
    payload = json.loads(buf.getvalue())
    assert payload["metrics"]["records_reconciled"] == 3


def test_sleep_finalize_promotion_is_idempotent(project):
    """F-1: re-running finalize never duplicate-inserts. A drained staging
    is a no-op; re-staging the same ids (a writer re-run after a crash)
    skips them as existing."""
    _stage(project, "rec_aaaaaaa1", summary="alpha")
    _stage(project, "rec_bbbbbbb2", summary="beta")

    # First cycle promotes both.
    c1 = _open_cycle(project)
    buf1 = io.StringIO()
    with redirect_stdout(buf1):
        rc1 = sleep_cli.cmd_sleep_finalize(_finalize_args(c1))
    assert rc1 == 0
    m1 = json.loads(buf1.getvalue())["metrics"]
    assert m1["records_reconciled"] == 2
    assert m1["records_reconcile_skipped_existing"] == 0

    # Second cycle with drained staging: nothing to do.
    c2 = _open_cycle(project)
    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        rc2 = sleep_cli.cmd_sleep_finalize(_finalize_args(c2))
    assert rc2 == 0
    m2 = json.loads(buf2.getvalue())["metrics"]
    assert m2["records_reconciled"] == 0
    assert m2["records_reconcile_skipped_existing"] == 0

    # Re-stage the SAME ids (writer re-run): skipped as existing, no dups,
    # and the stale staged rows are cleared.
    _stage(project, "rec_aaaaaaa1", summary="alpha")
    _stage(project, "rec_bbbbbbb2", summary="beta")
    c3 = _open_cycle(project)
    buf3 = io.StringIO()
    with redirect_stdout(buf3):
        rc3 = sleep_cli.cmd_sleep_finalize(_finalize_args(c3))
    assert rc3 == 0
    m3 = json.loads(buf3.getvalue())["metrics"]
    assert m3["records_reconciled"] == 0
    assert m3["records_reconcile_skipped_existing"] == 2

    conn = connect(project / "storage" / "graph.db")
    try:
        repo = GraphRepo(conn)
        all_records = repo.list_records()
        staged_left = repo.list_staged()
    finally:
        conn.close()
    assert len(all_records) == 2
    assert staged_left == []


def test_sleep_finalize_empty_staging_is_safe(project):
    """Nothing staged → promotion is a no-op, cycle still completes."""
    cycle_id = _open_cycle(project)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = sleep_cli.cmd_sleep_finalize(_finalize_args(cycle_id))
    assert rc == 0
    m = json.loads(buf.getvalue())["metrics"]
    assert m["records_reconciled"] == 0
    assert m["records_reconcile_skipped_malformed"] == 0


def test_sleep_finalize_records_with_null_anchors_promote(project):
    """F-1: chunk-level record (NULL anchor offsets) promotes cleanly."""
    _stage(
        project, "rec_aaaaaaa1",
        anchor_start=None, anchor_end=None, summary="whole-chunk record",
    )
    cycle_id = _open_cycle(project)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = sleep_cli.cmd_sleep_finalize(_finalize_args(cycle_id))
    assert rc == 0
    conn = connect(project / "storage" / "graph.db")
    try:
        rec = GraphRepo(conn).get_record("rec_aaaaaaa1")
    finally:
        conn.close()
    assert rec is not None
    assert rec.anchor_offset_start is None
    assert rec.anchor_offset_end is None
    assert rec.summary == "whole-chunk record"


# -- D1/D2/D3: vec_index dual-write integration --------------------------


class _StubVecIndex:
    def __init__(self, raise_add: Exception | None = None):
        self.added: list[tuple[str, str]] = []
        self.deleted: list[str] = []
        self.raise_add = raise_add

    def __call__(self, *args, **kwargs):
        return self

    def add_record(self, rid: str, summary: str) -> None:
        if self.raise_add is not None:
            raise self.raise_add
        self.added.append((rid, summary))

    def delete_record(self, rid: str) -> None:
        self.deleted.append(rid)

    def has_record(self, rid: str) -> bool:
        return any(r == rid for r, _ in self.added)


def test_sleep_finalize_writes_to_vec_index(project):
    """D1: newly-promoted records are pushed into the vec_index."""
    _stage(project, "rec_aaaaaaa1", summary="alpha")
    _stage(project, "rec_bbbbbbb2", summary="beta")

    stub = _StubVecIndex()
    cycle_id = _open_cycle(project)
    args = _finalize_args(cycle_id, skip_vec_index=False)
    with patch.object(sleep_cli, "RecordsVecIndex", stub):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = sleep_cli.cmd_sleep_finalize(args)
    assert rc == 0
    # Both records pushed.
    rids = {rid for rid, _ in stub.added}
    assert rids == {"rec_aaaaaaa1", "rec_bbbbbbb2"}
    metrics = json.loads(buf.getvalue())["metrics"]
    assert metrics["vec_index_added"] == 2
    assert metrics["vec_index_already_present"] == 0
    assert metrics["vec_index_errors"] == []


def test_sleep_finalize_vec_index_failure_continues(project):
    """D2: vec_index.add_record raising for a record must NOT crash the
    cycle. Error is logged into metrics; SQLite write already succeeded;
    exit code stays 0."""
    _stage(project, "rec_aaaaaaa1", summary="alpha")

    stub = _StubVecIndex(raise_add=RuntimeError("vec down"))
    cycle_id = _open_cycle(project)
    args = _finalize_args(cycle_id, skip_vec_index=False)
    with patch.object(sleep_cli, "RecordsVecIndex", stub):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = sleep_cli.cmd_sleep_finalize(args)
    assert rc == 0  # partial success
    metrics = json.loads(buf.getvalue())["metrics"]
    assert metrics["vec_index_added"] == 0
    assert metrics["vec_index_errors"]
    err = metrics["vec_index_errors"][0]
    assert err["record_id"] == "rec_aaaaaaa1"
    assert "vec down" in err["error"]
    # SQLite write succeeded.
    conn = connect(project / "storage" / "graph.db")
    try:
        rec = GraphRepo(conn).get_record("rec_aaaaaaa1")
    finally:
        conn.close()
    assert rec is not None


# -- piece3: index_card promotion in sleep-finalize ----------------------


def _stage_card(
    project_root: Path, rid: str, *,
    doc_key: str,
    source: str = "file:///C:/papers/x.md",
    content_hash: str | None = "hash-v1",
    created_at: str = "2026-05-25T10:00:00Z",
    summary: str = "summary\n\nkey points\n\nrelevance",
) -> None:
    _stage(
        project_root, rid,
        source_uri=source, anchor_start=0, anchor_end=0,
        created_at=created_at, summary=summary,
        kind="index_card", doc_key=doc_key, source=source,
        content_hash=content_hash,
    )


def test_finalize_creates_index_card(project):
    """A new staged index_card is promoted and tagged kind='index_card'."""
    _stage_card(project, "rec_card0001", doc_key="doi:10.1/abc")
    cycle_id = _open_cycle(project)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = sleep_cli.cmd_sleep_finalize(_finalize_args(cycle_id))
    assert rc == 0
    m = json.loads(buf.getvalue())["metrics"]
    assert m["index_cards_created"] == 1
    assert m["records_reconciled"] == 1

    conn = connect(project / "storage" / "graph.db")
    try:
        repo = GraphRepo(conn)
        card = repo.get_record_by_doc_key("doi:10.1/abc")
    finally:
        conn.close()
    assert card is not None
    assert card.id == "rec_card0001"
    assert card.kind == "index_card"
    assert card.content_hash == "hash-v1"


def test_finalize_index_card_unchanged_is_skipped(project):
    """Same doc_key + same content_hash on a second cycle → skipped, no dup."""
    _stage_card(project, "rec_card0001", doc_key="doi:10.1/abc",
                content_hash="h1")
    c1 = _open_cycle(project)
    with redirect_stdout(io.StringIO()):
        sleep_cli.cmd_sleep_finalize(_finalize_args(c1))

    # Second cycle: re-stage a card with the SAME doc_key + hash (new id).
    _stage_card(project, "rec_card0002", doc_key="doi:10.1/abc",
                content_hash="h1")
    c2 = _open_cycle(project)
    buf = io.StringIO()
    with redirect_stdout(buf):
        sleep_cli.cmd_sleep_finalize(_finalize_args(c2))
    m = json.loads(buf.getvalue())["metrics"]
    # Nothing created; the staged duplicate cleared; DB keeps one card.
    assert m["index_cards_created"] == 0
    assert m["index_cards_unchanged"] == 1

    conn = connect(project / "storage" / "graph.db")
    try:
        repo = GraphRepo(conn)
        cards = [r for r in repo.list_records() if r.kind == "index_card"]
        staged_left = repo.list_staged()
    finally:
        conn.close()
    assert len(cards) == 1
    assert cards[0].id == "rec_card0001"  # original kept
    assert staged_left == []


def test_finalize_index_card_replaced_on_hash_change(project):
    """Same doc_key + different content_hash → old row dropped, new inserted,
    old id queued for vec deletion."""
    _stage_card(project, "rec_card0001", doc_key="doi:10.1/abc",
                content_hash="h1", summary="old summary")
    c1 = _open_cycle(project)
    with redirect_stdout(io.StringIO()):
        sleep_cli.cmd_sleep_finalize(_finalize_args(c1))

    # Regenerate: same doc_key, NEW hash, new id, new body.
    _stage_card(project, "rec_card0002", doc_key="doi:10.1/abc",
                content_hash="h2", summary="new summary")

    stub = _StubVecIndex()
    c2 = _open_cycle(project)
    buf = io.StringIO()
    with patch.object(sleep_cli, "RecordsVecIndex", stub):
        with redirect_stdout(buf):
            rc = sleep_cli.cmd_sleep_finalize(
                _finalize_args(c2, skip_vec_index=False),
            )
    assert rc == 0
    m = json.loads(buf.getvalue())["metrics"]
    assert m["index_cards_replaced"] == 1
    assert m["vec_index_deleted"] == 1

    # Exactly one card for the doc_key, now the new one.
    conn = connect(project / "storage" / "graph.db")
    try:
        repo = GraphRepo(conn)
        card = repo.get_record_by_doc_key("doi:10.1/abc")
        cards = [r for r in repo.list_records() if r.kind == "index_card"]
    finally:
        conn.close()
    assert len(cards) == 1
    assert card.id == "rec_card0002"
    assert card.content_hash == "h2"
    assert card.summary.startswith("new summary")
    # vec: old id deleted, new id added.
    assert "rec_card0001" in stub.deleted
    assert "rec_card0002" in {rid for rid, _ in stub.added}


def test_finalize_index_card_same_id_replace_reembeds(project):
    """Regression: a regenerated card that REUSES its id (same id, new
    content_hash) must still re-embed. The vec entry is deleted then
    re-added — otherwise the append-only ``has_record`` guard would skip
    the add and leave a stale embedding."""
    _stage_card(project, "rec_card0001", doc_key="doi:10.1/abc",
                content_hash="h1", summary="old body")
    c1 = _open_cycle(project)
    stub = _StubVecIndex()
    with patch.object(sleep_cli, "RecordsVecIndex", stub):
        with redirect_stdout(io.StringIO()):
            sleep_cli.cmd_sleep_finalize(_finalize_args(c1, skip_vec_index=False))
    assert "rec_card0001" in {rid for rid, _ in stub.added}

    # Re-stage the SAME id with new hash + body (id reuse).
    _stage_card(project, "rec_card0001", doc_key="doi:10.1/abc",
                content_hash="h2", summary="new body")
    c2 = _open_cycle(project)
    stub2 = _StubVecIndex()
    buf = io.StringIO()
    with patch.object(sleep_cli, "RecordsVecIndex", stub2):
        with redirect_stdout(buf):
            rc = sleep_cli.cmd_sleep_finalize(_finalize_args(c2, skip_vec_index=False))
    assert rc == 0
    m = json.loads(buf.getvalue())["metrics"]
    assert m["index_cards_replaced"] == 1
    # Deleted then re-added under the same id -> embedding refreshed.
    assert "rec_card0001" in stub2.deleted
    assert "rec_card0001" in {rid for rid, _ in stub2.added}

    conn = connect(project / "storage" / "graph.db")
    try:
        card = GraphRepo(conn).get_record_by_doc_key("doi:10.1/abc")
    finally:
        conn.close()
    assert card.content_hash == "h2"
    assert card.summary.startswith("new body")


def test_finalize_index_card_without_doc_key_is_malformed(project):
    """An index_card with no doc_key is malformed — never promoted; the
    staged row moves to trash (with a reason) so it can't haunt later
    cycles."""
    _stage(
        project, "rec_nokey001",
        source_uri="file:///a.md", anchor_start=0, anchor_end=0,
        summary="body without doc_key", kind="index_card",
    )
    cycle_id = _open_cycle(project)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = sleep_cli.cmd_sleep_finalize(_finalize_args(cycle_id))
    assert rc == 0
    m = json.loads(buf.getvalue())["metrics"]
    assert m["records_reconcile_skipped_malformed"] == 1
    assert m["index_cards_created"] == 0
    conn = connect(project / "storage" / "graph.db")
    try:
        repo = GraphRepo(conn)
        assert repo.get_staged("rec_nokey001") is None
        assert repo.get_trashed("rec_nokey001") is not None
    finally:
        conn.close()


def test_finalize_dedups_index_cards_within_cycle(project):
    """Two staged cards with the same doc_key in ONE cycle → one card kept
    (stage_record replaces the prior same-key staged card)."""
    _stage_card(project, "rec_card0001", doc_key="doi:10.1/abc",
                content_hash="h1")
    _stage_card(project, "rec_card0002", doc_key="doi:10.1/abc",
                content_hash="h1")
    cycle_id = _open_cycle(project)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = sleep_cli.cmd_sleep_finalize(_finalize_args(cycle_id))
    assert rc == 0
    conn = connect(project / "storage" / "graph.db")
    try:
        cards = [r for r in GraphRepo(conn).list_records()
                 if r.kind == "index_card"]
    finally:
        conn.close()
    assert len(cards) == 1  # one of the two; the other deduped


def test_finalize_claims_and_cards_coexist(project):
    """A mixed cycle: claims keep the INSERT-OR-IGNORE path, cards the
    upsert path; claims carry no doc fields."""
    _stage(project, "rec_claim0001", summary="a claim")
    _stage_card(project, "rec_card0001", doc_key="doi:10.1/abc")
    cycle_id = _open_cycle(project)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = sleep_cli.cmd_sleep_finalize(_finalize_args(cycle_id))
    assert rc == 0
    conn = connect(project / "storage" / "graph.db")
    try:
        repo = GraphRepo(conn)
        claim = repo.get_record("rec_claim0001")
        card = repo.get_record("rec_card0001")
    finally:
        conn.close()
    assert claim.kind == "claim"
    assert claim.doc_key is None and claim.content_hash is None
    assert card.kind == "index_card"
    assert card.doc_key == "doi:10.1/abc"


def test_finalize_claim_with_doc_ref_and_provisional_stub(project):
    """piece3-B: a staged claim carrying a doc_key + title (a doc reference)
    promotes with those fields; a provisional stub card promotes as an
    index_card; get_record_by_doc_key returns the CARD, not the claim."""
    # claim referencing a doc (kind=claim; carries doc_key+title)
    _stage(
        project, "rec_refclaim",
        source_uri="qmd://priming-stream-imports/c/x.md",
        anchor_start=0, anchor_end=5,
        created_at="2026-06-01T00:00:00Z",
        summary="We treat D=2.71 as method-specific, not natural-space.",
        doc_key="t:delgiudice-2012-personality", title="Del Giudice 2012",
    )
    # provisional stub card for the same doc
    _stage(
        project, "rec_stubcard",
        source_uri="doc://t:delgiudice-2012-personality",
        anchor_start=0, anchor_end=0,
        created_at="2026-06-01T00:00:00Z",
        summary="## Summary\nDel Giudice 2012 reports Mahalanobis D=2.71 on 16PF. [unverified]",
        kind="index_card", doc_key="t:delgiudice-2012-personality",
        title="Del Giudice 2012", provisional=True,
    )
    cycle_id = _open_cycle(project)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = sleep_cli.cmd_sleep_finalize(_finalize_args(cycle_id))
    assert rc == 0
    m = json.loads(buf.getvalue())["metrics"]
    assert m["index_cards_created"] == 1
    assert m["records_reconciled"] == 2  # claim + card

    conn = connect(project / "storage" / "graph.db")
    try:
        repo = GraphRepo(conn)
        claim = repo.get_record("rec_refclaim")
        card = repo.get_record_by_doc_key("t:delgiudice-2012-personality")
    finally:
        conn.close()
    # claim carries the doc reference, stays a claim
    assert claim.kind == "claim"
    assert claim.doc_key == "t:delgiudice-2012-personality"
    assert claim.title == "Del Giudice 2012"
    # the card is the document node — provisional stub, not the claim
    assert card.kind == "index_card"
    assert card.id == "rec_stubcard"
    assert card.provisional is True
    assert card.content_hash is None


# -- registered with main parser -----------------------------------------


def test_sleep_subcommands_registered():
    """``sleep-prepare`` + ``sleep-finalize`` parse via the top-level
    ``Priming Stream`` parser (smoke; catches missed register call)."""
    from priming_stream.cli.main import _build_parser
    parser = _build_parser()
    sub_action = next(
        a for a in parser._actions
        if isinstance(a, argparse._SubParsersAction)
    )
    assert "sleep-prepare" in sub_action.choices
    assert "sleep-finalize" in sub_action.choices


# -- cursor crash-safety tests (P1-B) ------------------------------------


def test_cursor_prepare_does_not_advance(project):
    """(a) prepare must NOT commit the cursor — crash between prepare and
    finalize leaves chunks re-extractable on the next run."""
    _seed_chunks(project, ["export_a_p0", "export_a_p1"])
    cursor_path = project / "storage" / "corpus" / "_cursor.json"

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = sleep_cli.cmd_sleep_prepare(_prep_args())
    assert rc == 0
    manifest = json.loads(buf.getvalue())
    assert len(manifest["prepared_chunks"]) == 2

    # Key assertion: cursor file must NOT exist after prepare.
    assert not cursor_path.exists()


def test_cursor_finalize_with_manifest_advances(project):
    """(b) finalize with a manifest commits the cursor to the last chunk path."""
    _seed_chunks(project, ["export_a_p0", "export_a_p1", "export_a_p2"])
    cursor_path = project / "storage" / "corpus" / "_cursor.json"
    manifest_path = project / "storage" / "corpus" / "_sleep_manifest.json"

    # Run prepare; write manifest to standard location.
    buf = io.StringIO()
    with redirect_stdout(buf):
        sleep_cli.cmd_sleep_prepare(_prep_args())
    manifest = json.loads(buf.getvalue())
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    # Finalize with explicit manifest path.
    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        rc = sleep_cli.cmd_sleep_finalize(
            _finalize_args(manifest["cycle_id"], manifest_path=str(manifest_path))
        )
    assert rc == 0

    # Cursor must now be set to the last chunk in the manifest.
    assert cursor_path.exists()
    state = json.loads(cursor_path.read_text(encoding="utf-8"))
    last_chunk_id = manifest["prepared_chunks"][-1]["chunk_id"]
    assert state["last_chunk_id"] == last_chunk_id


def test_cursor_finalize_without_manifest_leaves_cursor_untouched(project):
    """(c) finalize without manifest / doc-only cycle → cursor untouched."""
    _seed_chunks(project, ["export_a_p0"])
    cursor_path = project / "storage" / "corpus" / "_cursor.json"
    manifest_path = project / "storage" / "corpus" / "_sleep_manifest.json"

    # No manifest on disk; finalize with manifest_path=None (default).
    conn = connect(project / "storage" / "graph.db")
    try:
        repo = GraphRepo(conn)
        cycle_id = repo.start_sleep_cycle(started_at="2026-06-10T00:00:00Z")
    finally:
        conn.close()

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = sleep_cli.cmd_sleep_finalize(_finalize_args(cycle_id, manifest_path=None))
    assert rc == 0

    # Neither cursor nor manifest file should have been created.
    assert not cursor_path.exists()
    assert not manifest_path.exists()


def test_cursor_finalize_cycle_mismatch_leaves_cursor_untouched(project, capsys):
    """A manifest from a DIFFERENT cycle must never advance this cycle's cursor.

    Scenario guarded: prepare(N+1) wrote a fresh manifest, then an operator
    re-runs ``finalize --cycle-id N`` — the N+1 chunks were not extracted yet,
    so committing the cursor from that manifest would silently lose them.
    """
    _seed_chunks(project, ["export_a_p0", "export_a_p1"])
    cursor_path = project / "storage" / "corpus" / "_cursor.json"
    manifest_path = project / "storage" / "corpus" / "_sleep_manifest.json"

    # prepare twice without finalizing: cycle 1, then cycle 2 — the second
    # overwrites the on-disk manifest, so the manifest now belongs to cycle 2.
    buf = io.StringIO()
    with redirect_stdout(buf):
        sleep_cli.cmd_sleep_prepare(_prep_args())
    old_cycle_id = json.loads(buf.getvalue())["cycle_id"]
    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        sleep_cli.cmd_sleep_prepare(_prep_args())
    assert json.loads(buf2.getvalue())["cycle_id"] != old_cycle_id

    # Operator mistakenly re-runs finalize for the OLD cycle.
    buf3 = io.StringIO()
    with redirect_stdout(buf3):
        rc = sleep_cli.cmd_sleep_finalize(
            _finalize_args(old_cycle_id, manifest_path=str(manifest_path))
        )
    assert rc == 0

    # Cursor untouched + a loud warning on stderr.
    assert not cursor_path.exists()
    assert "cursor not advanced" in capsys.readouterr().err


def test_cursor_prepare_crash_reruns_same_chunks(project):
    """(d) prepare → crash (no finalize) → re-prepare returns identical chunks.

    This is the idempotency guarantee: since the cursor did not advance, the
    second prepare sees the same pending chunks and re-materialises them
    (idempotent overwrite on disk), so no data is lost.
    """
    _seed_chunks(project, ["export_a_p0", "export_a_p1"])

    # First prepare (simulated crash: no finalize follows).
    buf1 = io.StringIO()
    with redirect_stdout(buf1):
        rc1 = sleep_cli.cmd_sleep_prepare(_prep_args())
    assert rc1 == 0
    m1 = json.loads(buf1.getvalue())
    assert len(m1["prepared_chunks"]) == 2

    # Second prepare (the "re-run after crash").
    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        rc2 = sleep_cli.cmd_sleep_prepare(_prep_args())
    assert rc2 == 0
    m2 = json.loads(buf2.getvalue())

    # Must return the same chunks — nothing was lost.
    assert {c["chunk_id"] for c in m2["prepared_chunks"]} == {
        c["chunk_id"] for c in m1["prepared_chunks"]
    }
    # All paths on disk (idempotent overwrite).
    for c in m2["prepared_chunks"]:
        assert Path(c["path"]).exists()
