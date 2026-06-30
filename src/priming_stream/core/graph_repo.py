"""Graph repository — sole writer of the v0.7-x SQLite database.

v0.7-x is records-as-substrate: only ``records`` and ``sleep_cycles``.
All legacy nodes/edges/decisions/aliases/record_mentions surfaces are
gone. Other modules read and mutate through this class.
"""
from __future__ import annotations

import sqlite3

from priming_stream.core.models import Record


class GraphRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    @staticmethod
    def _record(row: sqlite3.Row) -> Record:
        # ``kind``/``doc_key``/``source``/``content_hash`` are piece3
        # columns. Read them via dict-style access with a guard so a row
        # from a hypothetical pre-piece3 connection (no such columns) still
        # builds a claim Record rather than raising. On any migrated DB the
        # columns are always present; the guard is belt-and-suspenders.
        keys = row.keys()
        return Record(
            id=row["id"],
            source_uri=row["source_uri"],
            anchor_offset_start=row["anchor_offset_start"],
            anchor_offset_end=row["anchor_offset_end"],
            summary=row["summary"],
            created_at=row["created_at"],
            kind=row["kind"] if "kind" in keys else "claim",
            doc_key=row["doc_key"] if "doc_key" in keys else None,
            source=row["source"] if "source" in keys else None,
            content_hash=row["content_hash"] if "content_hash" in keys else None,
            title=row["title"] if "title" in keys else None,
            provisional=bool(row["provisional"]) if "provisional" in keys else False,
            source_date=row["source_date"] if "source_date" in keys else None,
        )

    # -- records ----------------------------------------------------------

    def create_record(self, record: Record) -> None:
        self.conn.execute(
            "INSERT INTO records "
            "(id, source_uri, anchor_offset_start, anchor_offset_end, "
            "summary, created_at, source_date, kind, doc_key, source, "
            "content_hash, title, provisional) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.id,
                record.source_uri,
                record.anchor_offset_start,
                record.anchor_offset_end,
                record.summary,
                record.created_at,
                record.source_date,
                record.kind,
                record.doc_key,
                record.source,
                record.content_hash,
                record.title,
                1 if record.provisional else 0,
            ),
        )
        self.conn.commit()

    def update_record_summary(self, record_id: str, new_summary: str) -> bool:
        """Replace a record's ``summary`` in place. Returns True iff a row
        matched. The ``records_au`` trigger keeps the FTS5 shadow in sync;
        the ChromaDB embedding is a SEPARATE, derived store — callers MUST
        re-embed (``RecordsVecIndex.add_record``, best-effort: Chroma is
        rebuildable from SQLite via ``vec-index-rebuild``)."""
        cur = self.conn.execute(
            "UPDATE records SET summary = ? WHERE id = ?",
            (new_summary, record_id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def get_record(self, record_id: str) -> Record | None:
        row = self.conn.execute(
            "SELECT * FROM records WHERE id = ?", (record_id,)
        ).fetchone()
        return self._record(row) if row else None

    def get_record_by_doc_key(self, doc_key: str) -> Record | None:
        """The single index_card for a ``doc_key`` (one-card-per-document
        invariant, enforced by the partial unique index). Returns ``None``
        if no document with this key has been carded yet. Used by the
        sleep-finalize reconcile to decide create-vs-replace-vs-skip.

        piece3-B: filters ``kind='index_card'`` because a *claim* may now
        carry the same ``doc_key`` as a (non-unique) reference to the
        document — only the card is the document node."""
        row = self.conn.execute(
            "SELECT * FROM records WHERE doc_key = ? AND kind = 'index_card'",
            (doc_key,),
        ).fetchone()
        return self._record(row) if row else None

    def card_exists_with_content_hash(self, content_hash: str) -> bool:
        """True iff some index_card already carries this ``content_hash``.

        piece3-C (canonical rewire): the explicit-ingest prefilter can no
        longer key on ``doc_key`` (that's now derived from document metadata the
        worker extracts, i.e. only known AFTER the worker runs). It prefilters
        on the source ``content_hash`` instead — an unchanged, already-carded
        document hashes the same, so we skip re-running the LLM worker on it."""
        if not content_hash:
            return False
        row = self.conn.execute(
            "SELECT 1 FROM records WHERE kind='index_card' AND content_hash = ? "
            "LIMIT 1",
            (content_hash,),
        ).fetchone()
        return row is not None

    def get_card_by_content_hash(self, content_hash: str) -> Record | None:
        """The index_card carrying this exact ``content_hash`` (the strongest,
        no-LLM same-document signal: identical file bytes). None if absent."""
        if not content_hash:
            return None
        row = self.conn.execute(
            "SELECT * FROM records WHERE kind='index_card' AND content_hash = ? "
            "LIMIT 1",
            (content_hash,),
        ).fetchone()
        return self._record(row) if row else None

    def delete_record(self, record_id: str) -> None:
        """Delete a record by id. Used when an index_card is regenerated
        (content_hash changed): the stale row is removed and a fresh one
        inserted. No-op if the id is absent."""
        self.conn.execute("DELETE FROM records WHERE id = ?", (record_id,))
        self.conn.commit()

    def replace_record(self, existing_id: str, new_record: Record) -> None:
        """DELETE existing_id and INSERT new_record in a single transaction.

        Atomicity: a crash between the two operations can no longer leave the
        database in a state where the old card is absent and the new one has
        not yet been written. FTS5 triggers (records_au / records_ad) fire as
        normal — they are DML-level, so they execute inside the transaction.
        Uses an explicit savepoint so the caller's connection state is
        preserved regardless of Python's implicit-transaction handling.
        """
        self.conn.execute("SAVEPOINT replace_record_sp")
        try:
            self.conn.execute(
                "DELETE FROM records WHERE id = ?", (existing_id,)
            )
            self.conn.execute(
                "INSERT INTO records "
                "(id, source_uri, anchor_offset_start, anchor_offset_end, "
                "summary, created_at, source_date, kind, doc_key, source, "
                "content_hash, title, provisional) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    new_record.id,
                    new_record.source_uri,
                    new_record.anchor_offset_start,
                    new_record.anchor_offset_end,
                    new_record.summary,
                    new_record.created_at,
                    new_record.source_date,
                    new_record.kind,
                    new_record.doc_key,
                    new_record.source,
                    new_record.content_hash,
                    new_record.title,
                    1 if new_record.provisional else 0,
                ),
            )
        except Exception:
            self.conn.execute("ROLLBACK TO SAVEPOINT replace_record_sp")
            self.conn.execute("RELEASE SAVEPOINT replace_record_sp")
            raise
        self.conn.execute("RELEASE SAVEPOINT replace_record_sp")
        self.conn.commit()

    # -- staging (SQL-canonical write-path) --------------------------------
    #
    # ``records_staging`` holds the ingest cycle's freshly-extracted records
    # BEFORE reconcile + finalize. It replaces the old on-disk boundary: a
    # record .md in ``records/`` whose id was not yet in SQLite used to mean
    # "incoming this cycle"; now table membership does. Staged rows are not
    # FTS-indexed and not embedded — invisible to every read surface until
    # promoted by sleep-finalize.

    _COLS = (
        "id, source_uri, anchor_offset_start, anchor_offset_end, summary, "
        "created_at, source_date, kind, doc_key, source, content_hash, "
        "title, provisional"
    )

    @staticmethod
    def _record_params(record: Record) -> tuple:
        return (
            record.id,
            record.source_uri,
            record.anchor_offset_start,
            record.anchor_offset_end,
            record.summary,
            record.created_at,
            record.source_date,
            record.kind,
            record.doc_key,
            record.source,
            record.content_hash,
            record.title,
            1 if record.provisional else 0,
        )

    def stage_record(self, record: Record) -> None:
        """INSERT OR REPLACE into staging — idempotent on bulk-writer re-runs
        (the rec_id pool is stable per assignment, so a re-run REPLACEs the
        same rows, the staging analog of overwriting the same ``.md`` file).

        For an index_card, any staged card with the same ``doc_key`` is
        dropped first — the staging analog of the old filename-keyed-by-
        doc_key overwrite ("at most one card per doc_key"), which a re-run's
        fresh rec_id would otherwise violate.
        """
        if record.kind == "index_card" and record.doc_key:
            self.conn.execute(
                "DELETE FROM records_staging "
                "WHERE kind='index_card' AND doc_key = ?",
                (record.doc_key,),
            )
        self.conn.execute(
            f"INSERT OR REPLACE INTO records_staging ({self._COLS}) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            self._record_params(record),
        )
        self.conn.commit()

    def get_staged(self, record_id: str) -> Record | None:
        row = self.conn.execute(
            "SELECT * FROM records_staging WHERE id = ?", (record_id,)
        ).fetchone()
        return self._record(row) if row else None

    def list_staged(self, kind: str | None = None) -> list[Record]:
        """Staged rows, stable id order (deterministic promote/plan order)."""
        if kind is None:
            rows = self.conn.execute(
                "SELECT * FROM records_staging ORDER BY id"
            )
        else:
            rows = self.conn.execute(
                "SELECT * FROM records_staging WHERE kind = ? ORDER BY id",
                (kind,),
            )
        return [self._record(r) for r in rows]

    def delete_staged(self, record_id: str) -> None:
        self.conn.execute(
            "DELETE FROM records_staging WHERE id = ?", (record_id,)
        )
        self.conn.commit()

    def repoint_staged_doc_key(self, old_key: str, new_key: str) -> int:
        """Re-point this cycle's staged CLAIMS off an absorbed card's key onto
        the survivor's (the staging analog of ``repoint_doc_key_in_mds``).
        Cards are left alone — the absorbed staged card is deleted by the
        caller. Returns the count repointed."""
        if old_key == new_key:
            return 0
        cur = self.conn.execute(
            "UPDATE records_staging SET doc_key = ? "
            "WHERE doc_key = ? AND kind = 'claim'",
            (new_key, old_key),
        )
        self.conn.commit()
        return cur.rowcount

    def promote_record(self, record: Record) -> None:
        """INSERT into ``records`` + DELETE from staging, one transaction —
        a crash can't both promote and re-promote a row. FTS5 trigger fires
        on the INSERT (DML-level, inside the transaction)."""
        self.conn.execute("SAVEPOINT promote_sp")
        try:
            self.conn.execute(
                f"INSERT INTO records ({self._COLS}) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                self._record_params(record),
            )
            self.conn.execute(
                "DELETE FROM records_staging WHERE id = ?", (record.id,)
            )
        except Exception:
            self.conn.execute("ROLLBACK TO SAVEPOINT promote_sp")
            self.conn.execute("RELEASE SAVEPOINT promote_sp")
            raise
        self.conn.execute("RELEASE SAVEPOINT promote_sp")
        self.conn.commit()

    # -- trash (soft delete) ------------------------------------------------

    def trash_record(self, record_id: str, reason: str | None = None) -> Record | None:
        """Soft-delete: move a row ``records`` → ``records_trash`` in one
        transaction (the ``records_ad`` trigger drops the FTS5 shadow).
        Returns the trashed Record, or None if the id is absent. The caller
        is responsible for the ChromaDB delete (vec is a derived cache)."""
        from priming_stream.core.models import now_iso
        rec = self.get_record(record_id)
        if rec is None:
            return None
        self.conn.execute("SAVEPOINT trash_sp")
        try:
            self.conn.execute(
                f"INSERT OR REPLACE INTO records_trash ({self._COLS}, "
                "deleted_at, delete_reason) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                self._record_params(rec) + (now_iso(), reason),
            )
            self.conn.execute(
                "DELETE FROM records WHERE id = ?", (record_id,)
            )
        except Exception:
            self.conn.execute("ROLLBACK TO SAVEPOINT trash_sp")
            self.conn.execute("RELEASE SAVEPOINT trash_sp")
            raise
        self.conn.execute("RELEASE SAVEPOINT trash_sp")
        self.conn.commit()
        return rec

    def trash_staged(self, record_id: str, reason: str | None = None) -> Record | None:
        """Soft-delete a STAGED row (this-cycle record killed by reconcile
        before it ever reached the substrate): staging → trash, one
        transaction. Nothing in ``records``/FTS/Chroma to remove. Returns the
        trashed Record or None."""
        from priming_stream.core.models import now_iso
        rec = self.get_staged(record_id)
        if rec is None:
            return None
        self.conn.execute("SAVEPOINT trash_staged_sp")
        try:
            self.conn.execute(
                f"INSERT OR REPLACE INTO records_trash ({self._COLS}, "
                "deleted_at, delete_reason) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                self._record_params(rec) + (now_iso(), reason),
            )
            self.conn.execute(
                "DELETE FROM records_staging WHERE id = ?", (record_id,)
            )
        except Exception:
            self.conn.execute("ROLLBACK TO SAVEPOINT trash_staged_sp")
            self.conn.execute("RELEASE SAVEPOINT trash_staged_sp")
            raise
        self.conn.execute("RELEASE SAVEPOINT trash_staged_sp")
        self.conn.commit()
        return rec

    def get_trashed(self, record_id: str) -> Record | None:
        row = self.conn.execute(
            "SELECT * FROM records_trash WHERE id = ?", (record_id,)
        ).fetchone()
        return self._record(row) if row else None

    def restore_record(self, record_id: str) -> Record | None:
        """Reverse a soft delete: trash → ``records``, one transaction.
        Returns the restored Record, or None if absent from trash (or its id
        is already live again). The caller re-embeds into ChromaDB."""
        rec = self.get_trashed(record_id)
        if rec is None or self.get_record(record_id) is not None:
            return None
        self.conn.execute("SAVEPOINT restore_sp")
        try:
            self.conn.execute(
                f"INSERT INTO records ({self._COLS}) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                self._record_params(rec),
            )
            self.conn.execute(
                "DELETE FROM records_trash WHERE id = ?", (record_id,)
            )
        except Exception:
            self.conn.execute("ROLLBACK TO SAVEPOINT restore_sp")
            self.conn.execute("RELEASE SAVEPOINT restore_sp")
            raise
        self.conn.execute("RELEASE SAVEPOINT restore_sp")
        self.conn.commit()
        return rec

    def list_records(self, limit: int | None = None) -> list[Record]:
        """Most recent first by ``created_at``; ties broken by id desc."""
        sql = "SELECT * FROM records ORDER BY created_at DESC, id DESC"
        params: tuple = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (int(limit),)
        return [self._record(r) for r in self.conn.execute(sql, params)]

    def records_by_source_uri(self, prefix: str) -> list[Record]:
        """Records whose ``source_uri`` starts with ``prefix``. Ordered by
        ``created_at`` descending (recent first)."""
        # Escape LIKE special chars in the prefix so a literal '_' or '%' in a
        # source_uri is not treated as a wildcard.
        escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        rows = self.conn.execute(
            "SELECT * FROM records WHERE source_uri LIKE ? ESCAPE '\\' "
            "ORDER BY created_at DESC, id DESC",
            (escaped + "%",),
        )
        return [self._record(r) for r in rows]

    # -- sleep cycles -----------------------------------------------------

    def start_sleep_cycle(self, *, started_at: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO sleep_cycles (started_at) VALUES (?)",
            (started_at,),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def finish_sleep_cycle(
        self,
        cycle_id: int,
        *,
        completed_at: str,
        chunks_materialized: int,
        records_created: int,
        records_skipped: int,
        metrics_json: str,
        notes: str | None,
    ) -> None:
        self.conn.execute(
            "UPDATE sleep_cycles SET completed_at = ?, "
            "chunks_materialized = ?, records_created = ?, "
            "records_skipped = ?, metrics_json = ?, notes = ? "
            "WHERE id = ?",
            (
                completed_at,
                chunks_materialized,
                records_created,
                records_skipped,
                metrics_json,
                notes,
                cycle_id,
            ),
        )
        self.conn.commit()

    def list_sleep_cycles(self, limit: int = 50) -> list[dict]:
        """Most recent first by id descending. Returns dict rows so callers
        can ignore the exact column set (audit-facing surface)."""
        rows = self.conn.execute(
            "SELECT id, started_at, completed_at, chunks_materialized, "
            "records_created, records_skipped, metrics_json, notes "
            "FROM sleep_cycles ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "started_at": r["started_at"],
                "completed_at": r["completed_at"],
                "chunks_materialized": r["chunks_materialized"],
                "records_created": r["records_created"],
                "records_skipped": r["records_skipped"],
                "metrics_json": r["metrics_json"],
                "notes": r["notes"],
            }
            for r in rows
        ]
