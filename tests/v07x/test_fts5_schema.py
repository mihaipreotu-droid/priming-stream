"""FTS5 schema + sync triggers + one-time backfill (v0.7-x-bridge-daemon W-A).

Acceptance from spec §2.A:
- A1: clean migration creates records_fts + 3 triggers (records_ai/ad/au).
- A2: INSERT INTO records auto-populates records_fts (records_ai).
- A3: UPDATE/DELETE on records keeps records_fts in sync (records_au/ad).
- A4: pre-existing rows (no FTS yet) get backfilled by apply_migrations;
       idempotent on re-run.
"""
from __future__ import annotations

import sqlite3

from priming_stream.core.db import connect
from priming_stream.core.models import Record, new_record_id, now_iso
from priming_stream.core.schema import apply_migrations


_RECORDS_COLUMNS = (
    "id", "source_uri", "anchor_offset_start", "anchor_offset_end",
    "summary", "created_at",
)


def _bare_records_schema(conn: sqlite3.Connection) -> None:
    """Pre-bridge-daemon schema: records + sleep_cycles, no FTS.

    Used by A4 to simulate a DB created before the FTS migration ran.
    """
    conn.executescript(
        """
        CREATE TABLE records (
          id TEXT PRIMARY KEY,
          source_uri TEXT NOT NULL,
          anchor_offset_start INTEGER,
          anchor_offset_end INTEGER,
          summary TEXT NOT NULL,
          created_at TIMESTAMP NOT NULL
        );
        CREATE TABLE sleep_cycles (
          id INTEGER PRIMARY KEY,
          started_at TIMESTAMP,
          completed_at TIMESTAMP,
          chunks_materialized INTEGER,
          records_created INTEGER,
          records_skipped INTEGER,
          metrics_json TEXT,
          notes TEXT
        );
        """
    )
    conn.commit()


def _insert_record(conn: sqlite3.Connection, summary: str) -> str:
    rec = Record(
        id=new_record_id(),
        source_uri="qmd://test/r.md",
        anchor_offset_start=0,
        anchor_offset_end=len(summary),
        summary=summary,
        created_at=now_iso(),
    )
    conn.execute(
        "INSERT INTO records "
        "(id, source_uri, anchor_offset_start, anchor_offset_end, "
        "summary, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (rec.id, rec.source_uri, rec.anchor_offset_start,
         rec.anchor_offset_end, rec.summary, rec.created_at),
    )
    conn.commit()
    return rec.id


# -- A1 -------------------------------------------------------------------


def test_a1_fts_table_and_triggers_exist(tmp_path):
    conn = connect(tmp_path / "graph.db")
    try:
        apply_migrations(conn)
        names = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type IN ('table', 'trigger')"
            )
        }
    finally:
        conn.close()
    assert "records_fts" in names
    assert "records_ai" in names
    assert "records_ad" in names
    assert "records_au" in names


# -- A2 -------------------------------------------------------------------


def test_a2_insert_into_records_populates_fts(tmp_path):
    conn = connect(tmp_path / "graph.db")
    try:
        apply_migrations(conn)
        _insert_record(conn, "spreading activation across the bridge")
        hits = list(conn.execute(
            "SELECT rowid FROM records_fts WHERE summary MATCH ?",
            ("spreading",),
        ))
    finally:
        conn.close()
    assert len(hits) == 1


def test_a2_match_returns_no_false_positive(tmp_path):
    conn = connect(tmp_path / "graph.db")
    try:
        apply_migrations(conn)
        _insert_record(conn, "spreading activation across the bridge")
        hits = list(conn.execute(
            "SELECT rowid FROM records_fts WHERE summary MATCH ?",
            ("nonexistent",),
        ))
    finally:
        conn.close()
    assert hits == []


# -- A3 -------------------------------------------------------------------


def test_a3_update_records_summary_resyncs_fts(tmp_path):
    conn = connect(tmp_path / "graph.db")
    try:
        apply_migrations(conn)
        rid = _insert_record(conn, "alpha beta gamma")

        old_hits = list(conn.execute(
            "SELECT rowid FROM records_fts WHERE summary MATCH ?",
            ("alpha",),
        ))
        assert len(old_hits) == 1

        conn.execute(
            "UPDATE records SET summary = ? WHERE id = ?",
            ("delta epsilon zeta", rid),
        )
        conn.commit()

        old_after = list(conn.execute(
            "SELECT rowid FROM records_fts WHERE summary MATCH ?",
            ("alpha",),
        ))
        new_after = list(conn.execute(
            "SELECT rowid FROM records_fts WHERE summary MATCH ?",
            ("epsilon",),
        ))
    finally:
        conn.close()
    assert old_after == []
    assert len(new_after) == 1


def test_a3_delete_records_removes_fts_row(tmp_path):
    conn = connect(tmp_path / "graph.db")
    try:
        apply_migrations(conn)
        rid = _insert_record(conn, "ephemeral content for delete test")

        present = list(conn.execute(
            "SELECT rowid FROM records_fts WHERE summary MATCH ?",
            ("ephemeral",),
        ))
        assert len(present) == 1

        conn.execute("DELETE FROM records WHERE id = ?", (rid,))
        conn.commit()

        after = list(conn.execute(
            "SELECT rowid FROM records_fts WHERE summary MATCH ?",
            ("ephemeral",),
        ))
    finally:
        conn.close()
    assert after == []


# -- A4 -------------------------------------------------------------------


def test_a4_backfill_populates_preexisting_records(tmp_path):
    db_path = tmp_path / "graph.db"
    pre_conn = connect(db_path)
    try:
        _bare_records_schema(pre_conn)
        for i in range(5):
            _insert_record(pre_conn, f"backfill probe phrase number {i}")
        precount = pre_conn.execute(
            "SELECT count(*) FROM records"
        ).fetchone()[0]
        assert precount == 5
    finally:
        pre_conn.close()

    # Now run the migration (introduces FTS + triggers + backfills the 5).
    conn = connect(db_path)
    try:
        apply_migrations(conn)
        fts_count = conn.execute(
            "SELECT count(*) FROM records_fts"
        ).fetchone()[0]
        hits = list(conn.execute(
            "SELECT rowid FROM records_fts WHERE summary MATCH ?",
            ("backfill",),
        ))
    finally:
        conn.close()
    assert fts_count == 5
    assert len(hits) == 5


def test_a4_backfill_is_idempotent(tmp_path):
    db_path = tmp_path / "graph.db"
    pre_conn = connect(db_path)
    try:
        _bare_records_schema(pre_conn)
        for i in range(5):
            _insert_record(pre_conn, f"idempotent probe {i}")
    finally:
        pre_conn.close()

    conn = connect(db_path)
    try:
        apply_migrations(conn)
        apply_migrations(conn)
        apply_migrations(conn)
        fts_count = conn.execute(
            "SELECT count(*) FROM records_fts"
        ).fetchone()[0]
    finally:
        conn.close()
    assert fts_count == 5
