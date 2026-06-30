"""Active-graph schema (v0.7-x). Minimal: records + sleep_cycles only.

Wipe-and-recreate semantics for the POC: this schema replaces the v0.7
fat schema (nodes/edges/decisions/aliases/record_mentions/...) wholesale.
No data migration. Drop ``storage/graph.db`` before first run.
"""
from __future__ import annotations

import sqlite3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
  id TEXT PRIMARY KEY,
  source_uri TEXT NOT NULL,
  anchor_offset_start INTEGER,
  anchor_offset_end INTEGER,
  summary TEXT NOT NULL,
  created_at TIMESTAMP NOT NULL,
  -- v0.7-x-B (sleep hygiene): the real conversation timestamp of the turn a
  -- record anchors to. ``created_at`` is the EXTRACTION date (uniform after a
  -- coldstart) and useless for recency/supersession; ``source_date`` is
  -- derived in Python from source_uri+anchor (core/source_date.py). NULL for
  -- index_cards / owner-authored records (no conversation date).
  source_date TEXT,
  -- v0.7-x-piece3 (document ingestion). OPTIONAL columns. ``kind`` is the
  -- one always-present discriminator; the rest carry values for index_card
  -- records, EXCEPT ``doc_key``/``title`` which a *claim* may also carry as a
  -- (non-unique) REFERENCE to a document it discusses (piece3-B). On a fresh
  -- DB these are created here; on an existing DB the ADD COLUMN backfill in
  -- apply_migrations() adds them (see _DOC_COLUMNS).
  kind TEXT NOT NULL DEFAULT 'claim',
  doc_key TEXT,            -- index_card: canonical identity (unique); claim: doc reference (non-unique)
  source TEXT,             -- index_card: location (path / URL / empty)
  content_hash TEXT,       -- index_card: source change-detection
  title TEXT,              -- document title (on a card, or on a claim referencing a doc) — keeps a DOI-keyed card findable
  provisional INTEGER NOT NULL DEFAULT 0  -- index_card: 1 = stub (no file yet, unverified), 0 = full / claim
);

CREATE INDEX IF NOT EXISTS idx_records_source_uri
  ON records(source_uri);
CREATE INDEX IF NOT EXISTS idx_records_created_at
  ON records(created_at);

CREATE TABLE IF NOT EXISTS sleep_cycles (
  id INTEGER PRIMARY KEY,
  started_at TIMESTAMP,
  completed_at TIMESTAMP,
  chunks_materialized INTEGER,
  records_created INTEGER,
  records_skipped INTEGER,
  metrics_json TEXT,
  notes TEXT
);

-- v0.7-x SQL-canonical: the ingest cycle's STAGING area. The bulk-writers
-- (writer.py / card_writer.py) insert this cycle's freshly-extracted records
-- here; reconcile reads "incoming" from here; sleep-finalize PROMOTES rows
-- into ``records`` (claims INSERT-OR-IGNORE by id, cards upsert by doc_key)
-- and clears them. Same columns as ``records``. Deliberately NO FTS5 shadow
-- and NO ChromaDB embedding until promotion — a staged record must never
-- surface in priming/search. This table replaces the old on-disk boundary
-- ("a .md in records/ whose id is not yet in SQLite" = incoming).
CREATE TABLE IF NOT EXISTS records_staging (
  id TEXT PRIMARY KEY,
  source_uri TEXT NOT NULL,
  anchor_offset_start INTEGER,
  anchor_offset_end INTEGER,
  summary TEXT NOT NULL,
  created_at TIMESTAMP NOT NULL,
  source_date TEXT,
  kind TEXT NOT NULL DEFAULT 'claim',
  doc_key TEXT,
  source TEXT,
  content_hash TEXT,
  title TEXT,
  provisional INTEGER NOT NULL DEFAULT 0
);

-- v0.7-x SQL-canonical: soft-delete trash. A deleted record's full row moves
-- here (owner ``record delete``, reconcile near-clone/contradiction deletes)
-- so the deletion stays reversible at zero cost — restore = move the row
-- back + re-embed. Replaces the old ``corpus/_deleted_records/*.md`` trash
-- (legacy files there remain as a historical archive). Not FTS-indexed, not
-- embedded — trashed records are invisible to every read surface.
CREATE TABLE IF NOT EXISTS records_trash (
  id TEXT PRIMARY KEY,
  source_uri TEXT NOT NULL,
  anchor_offset_start INTEGER,
  anchor_offset_end INTEGER,
  summary TEXT NOT NULL,
  created_at TIMESTAMP NOT NULL,
  source_date TEXT,
  kind TEXT NOT NULL DEFAULT 'claim',
  doc_key TEXT,
  source TEXT,
  content_hash TEXT,
  title TEXT,
  provisional INTEGER NOT NULL DEFAULT 0,
  deleted_at TIMESTAMP NOT NULL,
  delete_reason TEXT
);

-- FTS5 lexical index over records.summary. External-content table:
-- physical storage lives in ``records``; the triggers below keep the
-- FTS shadow in sync on INSERT/DELETE/UPDATE. Used by the bridge hook's
-- cold-path lexical fallback (v0.7-x-bridge-daemon).
CREATE VIRTUAL TABLE IF NOT EXISTS records_fts USING fts5(
  summary,
  content='records',
  content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS records_ai AFTER INSERT ON records BEGIN
  INSERT INTO records_fts(rowid, summary) VALUES (new.rowid, new.summary);
END;
CREATE TRIGGER IF NOT EXISTS records_ad AFTER DELETE ON records BEGIN
  INSERT INTO records_fts(records_fts, rowid, summary) VALUES('delete', old.rowid, old.summary);
END;
CREATE TRIGGER IF NOT EXISTS records_au AFTER UPDATE ON records BEGIN
  INSERT INTO records_fts(records_fts, rowid, summary) VALUES('delete', old.rowid, old.summary);
  INSERT INTO records_fts(rowid, summary) VALUES (new.rowid, new.summary);
END;
"""

# v0.7-x-piece3: optional index_card columns added to ``records``. Keyed
# by column name -> the ADD COLUMN DDL used to backfill a pre-piece3 DB.
# A fresh DB already has these (they are in _SCHEMA's CREATE TABLE); the
# backfill is a no-op there. Order is irrelevant for ADD COLUMN but the
# WHOLE backfill MUST run before the partial unique index on ``doc_key``
# is created (the index references a column that may not exist yet on an
# old DB). ``kind`` carries a NOT NULL DEFAULT so existing rows become
# 'claim' rows without a separate UPDATE.
_DOC_COLUMNS = {
    "kind": "kind TEXT NOT NULL DEFAULT 'claim'",
    "doc_key": "doc_key TEXT",
    "source": "source TEXT",
    "content_hash": "content_hash TEXT",
    "title": "title TEXT",
    "provisional": "provisional INTEGER NOT NULL DEFAULT 0",
}


def _backfill_doc_columns(conn: sqlite3.Connection) -> None:
    """Idempotently ADD COLUMN any piece3 columns missing from ``records``.

    Additive migration for an existing v0.7-x DB (the live substrate carries
    244 claim records on the pre-piece3 6-column schema). ``ALTER TABLE ADD
    COLUMN`` is cheap and preserves all existing rows; the NOT NULL DEFAULT
    on ``kind`` backfills existing rows to 'claim'.
    """
    existing = {r[1] for r in conn.execute("PRAGMA table_info(records)")}
    for name, ddl in _DOC_COLUMNS.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE records ADD COLUMN {ddl}")


def apply_migrations(conn: sqlite3.Connection) -> None:
    # Enable Write-Ahead Logging — permits concurrent readers + one writer
    # without lock contention. Needed because the bridge hook (read-only)
    # and the sleep cycle (writer) can run concurrently when /prime-sleep is
    # invoked while a CC session is active. journal_mode is persisted in
    # the DB header, so this only needs to fire once (subsequent calls are
    # no-op fast-path); kept here so a fresh ``prime init`` lands in WAL.
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(_SCHEMA)
    # v0.7-x-piece3: ensure the optional index_card columns exist on an
    # older DB BEFORE creating the partial unique index that references
    # ``doc_key``. On a fresh DB this is a no-op (columns are in _SCHEMA).
    _backfill_doc_columns(conn)
    # v0.7-x-B: source_date column (additive). On a fresh DB it is already in
    # _SCHEMA's CREATE TABLE; on an existing DB this ADD COLUMN backfills it
    # (NULL for every row until the one-shot backfill / sleep-finalize derives
    # it). See core/source_date.py.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(records)")}
    if "source_date" not in cols:
        conn.execute("ALTER TABLE records ADD COLUMN source_date TEXT")
    # One index card per document: enforce uniqueness of ``doc_key`` over
    # INDEX CARDS only. piece3-B: a claim may now carry ``doc_key`` as a
    # (non-unique) reference to a document it discusses, so the uniqueness
    # predicate is ``kind='index_card'`` — NOT ``doc_key IS NOT NULL`` (which
    # would reject a claim sharing a card's key). The index is renamed so the
    # DROP-old + CREATE-new(named) pair is idempotent across connects (the old
    # ``idx_records_doc_key`` is dropped if present, a no-op afterward).
    conn.execute("DROP INDEX IF EXISTS idx_records_doc_key")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_records_card_doc_key "
        "ON records(doc_key) WHERE kind='index_card'"
    )
    # One-time idempotent backfill of FTS5 over any pre-existing records.
    # The triggers above keep records_fts in sync for new
    # INSERT/UPDATE/DELETE; this handles rows that existed before the FTS
    # migration ran. We use FTS5's canonical ``rebuild`` command — the
    # naive ``INSERT INTO records_fts(rowid, summary) SELECT ... FROM
    # records`` does NOT tokenize on external-content FTS tables
    # (verified empirically: docsize stays empty and MATCH returns []).
    # The guard uses ``records_fts_docsize`` (FTS5 internal table that
    # tracks tokenized doc count) rather than ``count(*) FROM
    # records_fts`` (which is a virtual projection from the content
    # table and equals records.count whether tokenized or not).
    # apply_migrations runs per-connect, so the guard avoids re-
    # tokenizing the whole corpus on every CLI / hook fire.
    records_count = conn.execute(
        "SELECT count(*) FROM records"
    ).fetchone()[0]
    fts_indexed = conn.execute(
        "SELECT count(*) FROM records_fts_docsize"
    ).fetchone()[0]
    if records_count != fts_indexed:
        conn.execute("INSERT INTO records_fts(records_fts) VALUES('rebuild')")
    conn.commit()
