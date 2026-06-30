"""v0.7-x schema: the records-domain tables (+ sqlite system tables).

SQL-canonical (2026-06-12): ``records`` is the substrate's source of truth;
``records_staging`` holds the ingest cycle's incoming rows pre-promotion;
``records_trash`` holds soft-deleted rows (reversible). Plus ``sleep_cycles``
(audit) and the FTS5 shadow over ``records``.
"""
from __future__ import annotations

from priming_stream.core.db import connect
from priming_stream.core.schema import apply_migrations


# SQLite auto-creates ``sqlite_sequence`` when a table uses AUTOINCREMENT,
# and ``sqlite_master`` (the schema catalog) is always present. The v0.7-x
# schema uses plain INTEGER PRIMARY KEY on ``sleep_cycles`` (no AUTOINCREMENT
# keyword), so sqlite_sequence is NOT expected — but we tolerate it if a
# future schema bump introduces it. The acceptance contract is: only our
# records-domain tables plus the FTS5 shadow over ``records`` (the staging
# and trash tables are deliberately NOT FTS-indexed — invisible to search).
_USER_TABLES = {"records", "sleep_cycles", "records_staging", "records_trash"}
_FTS_TABLES = {
    "records_fts",
    "records_fts_data",
    "records_fts_idx",
    "records_fts_docsize",
    "records_fts_config",
}
_SYSTEM_PREFIXES = ("sqlite_",)


def _user_tables(conn) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    return {
        r[0] for r in rows
        if not r[0].startswith(_SYSTEM_PREFIXES)
    }


def test_only_two_tables(tmp_path):
    conn = connect(tmp_path / "graph.db")
    try:
        apply_migrations(conn)
        assert _user_tables(conn) == _USER_TABLES | _FTS_TABLES
    finally:
        conn.close()


def test_apply_migrations_idempotent(tmp_path):
    conn = connect(tmp_path / "graph.db")
    try:
        apply_migrations(conn)
        apply_migrations(conn)
        assert _user_tables(conn) == _USER_TABLES | _FTS_TABLES
    finally:
        conn.close()


def test_records_indexes_exist(tmp_path):
    conn = connect(tmp_path / "graph.db")
    try:
        apply_migrations(conn)
        idx = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index' "
                "AND tbl_name = 'records'"
            )
        }
    finally:
        conn.close()
    assert "idx_records_source_uri" in idx
    assert "idx_records_created_at" in idx
    # piece3-B: partial unique index enforcing one index_card per doc_key
    # (predicate kind='index_card', so claims may share a doc_key as a ref).
    assert "idx_records_card_doc_key" in idx
    # old predicate index must be gone (renamed/repredicated).
    assert "idx_records_doc_key" not in idx


def test_records_table_columns(tmp_path):
    conn = connect(tmp_path / "graph.db")
    try:
        apply_migrations(conn)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(records)")]
    finally:
        conn.close()
    assert cols == [
        "id",
        "source_uri",
        "anchor_offset_start",
        "anchor_offset_end",
        "summary",
        "created_at",
        # v0.7-x-B: derived conversation timestamp (NULL for cards / owner
        # records). On a FRESH DB it lands here (in CREATE TABLE order); an
        # ALTER-TABLE-migrated DB appends it last — order is cosmetic since
        # all access is by column name.
        "source_date",
        # piece3: optional index_card columns (claims leave them at
        # kind='claim' / NULL).
        "kind",
        "doc_key",
        "source",
        "content_hash",
        "title",
        "provisional",
    ]


def test_records_doc_columns_default_to_claim(tmp_path):
    """A row inserted without the piece3 columns is a claim with NULL
    doc fields — claims must never be expected to carry index_card data."""
    conn = connect(tmp_path / "graph.db")
    try:
        apply_migrations(conn)
        conn.execute(
            "INSERT INTO records (id, source_uri, summary, created_at) "
            "VALUES ('rec_claim01', 'qmd://priming-stream-imports/a.md', "
            "'a claim', '2026-05-01T00:00:00Z')"
        )
        conn.commit()
        row = conn.execute(
            "SELECT kind, doc_key, source, content_hash FROM records "
            "WHERE id = 'rec_claim01'"
        ).fetchone()
    finally:
        conn.close()
    assert row[0] == "claim"
    assert row[1] is None
    assert row[2] is None
    assert row[3] is None


def test_doc_key_unique_for_index_cards(tmp_path):
    """The partial unique index forbids two index_cards sharing a doc_key,
    while many claims (doc_key NULL) coexist freely."""
    import sqlite3

    conn = connect(tmp_path / "graph.db")
    try:
        apply_migrations(conn)
        # many NULL doc_keys are fine
        for i in range(3):
            conn.execute(
                "INSERT INTO records (id, source_uri, summary, created_at) "
                f"VALUES ('rec_c{i}', 'qmd://x/y.md', 'c', '2026-01-01T00:00:00Z')"
            )
        conn.execute(
            "INSERT INTO records (id, source_uri, summary, created_at, kind, doc_key) "
            "VALUES ('rec_d1', 'file:///a.md', 'card', '2026-01-01T00:00:00Z', "
            "'index_card', 'doi:10.1/x')"
        )
        conn.commit()
        raised = False
        try:
            conn.execute(
                "INSERT INTO records (id, source_uri, summary, created_at, kind, doc_key) "
                "VALUES ('rec_d2', 'file:///b.md', 'card2', '2026-01-01T00:00:00Z', "
                "'index_card', 'doi:10.1/x')"
            )
            conn.commit()
        except sqlite3.IntegrityError:
            raised = True

        # piece3-B: a CLAIM may share the card's doc_key (non-unique ref) —
        # the unique index is scoped to kind='index_card', so this is allowed.
        claim_ref_ok = True
        try:
            conn.execute(
                "INSERT INTO records (id, source_uri, summary, created_at, kind, doc_key) "
                "VALUES ('rec_ref1', 'qmd://x/y.md', 'a claim about that doc', "
                "'2026-01-01T00:00:00Z', 'claim', 'doi:10.1/x')"
            )
            conn.commit()
        except sqlite3.IntegrityError:
            claim_ref_ok = False
    finally:
        conn.close()
    assert raised, "duplicate doc_key on two index_cards must violate the unique index"
    assert claim_ref_ok, "a claim sharing a card's doc_key (as a ref) must be allowed"


def test_sleep_cycles_table_columns(tmp_path):
    conn = connect(tmp_path / "graph.db")
    try:
        apply_migrations(conn)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(sleep_cycles)")]
    finally:
        conn.close()
    assert cols == [
        "id",
        "started_at",
        "completed_at",
        "chunks_materialized",
        "records_created",
        "records_skipped",
        "metrics_json",
        "notes",
    ]


_RECORD_COLS = [
    "id", "source_uri", "anchor_offset_start", "anchor_offset_end",
    "summary", "created_at", "source_date", "kind", "doc_key", "source",
    "content_hash", "title", "provisional",
]


def test_staging_table_mirrors_records_columns(tmp_path):
    conn = connect(tmp_path / "graph.db")
    try:
        apply_migrations(conn)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(records_staging)")]
    finally:
        conn.close()
    assert cols == _RECORD_COLS


def test_trash_table_is_records_plus_deletion_metadata(tmp_path):
    conn = connect(tmp_path / "graph.db")
    try:
        apply_migrations(conn)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(records_trash)")]
    finally:
        conn.close()
    assert cols == _RECORD_COLS + ["deleted_at", "delete_reason"]


def test_staging_and_trash_not_fts_indexed(tmp_path):
    """A staged or trashed row must never surface via the FTS5 shadow —
    the triggers fire on ``records`` only."""
    conn = connect(tmp_path / "graph.db")
    try:
        apply_migrations(conn)
        conn.execute(
            "INSERT INTO records_staging (id, source_uri, summary, created_at) "
            "VALUES ('rec_stg1', 'qmd://x/y.md', 'zebra xylophone staged', "
            "'2026-01-01T00:00:00Z')"
        )
        conn.execute(
            "INSERT INTO records_trash (id, source_uri, summary, created_at, "
            "deleted_at) VALUES ('rec_tr1', 'qmd://x/y.md', "
            "'zebra xylophone trashed', '2026-01-01T00:00:00Z', "
            "'2026-01-02T00:00:00Z')"
        )
        conn.commit()
        hits = conn.execute(
            "SELECT rowid FROM records_fts WHERE records_fts MATCH 'zebra'"
        ).fetchall()
    finally:
        conn.close()
    assert hits == []
