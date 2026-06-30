"""``prime vec-index-rebuild`` — wipe + repopulate the ChromaDB records index.

SQL-canonical (2026-06-12): the ``records`` table in ``graph.db`` is the
source of truth; ChromaDB is a derived index. This subcommand drops the
existing ``records`` collection, SELECTs every (id, summary) row from
SQLite, and bulk-upserts in batches of 50 via :class:`RecordsVecIndex`.

Staged rows (``records_staging``) are deliberately NOT embedded — a record
enters the index only when sleep-finalize promotes it. Trash rows likewise.

On startup, if a legacy ``storage/qmd-corpus/`` exists and the new
``storage/corpus/`` doesn't, we rename atomically (idempotent helper in
``priming_stream.core.paths``) before opening. This means existing installs
upgrade transparently the first time the rebuild runs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from priming_stream.core.config import load_config
from priming_stream.core.paths import resolve_paths, migrate_qmd_corpus_to_corpus
from priming_stream.integrations.vec_index import RecordsVecIndex


_BATCH_SIZE = 50


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "vec-index-rebuild",
        help="drop + repopulate the ChromaDB 'records' collection from the "
             "SQLite records table",
    )
    p.add_argument(
        "--persist-dir", default=None,
        help="ChromaDB persist dir (default: from config.vec_index.persist_dir)",
    )
    p.add_argument(
        "--db-path", default=None,
        help="SQLite source database (default: <storage>/graph.db)",
    )
    p.set_defaults(func=_cmd_vec_index_rebuild)


def _cmd_vec_index_rebuild(args: argparse.Namespace) -> int:
    try:
        cfg = load_config()
        paths = resolve_paths(cfg, project_root=Path.cwd())

        # v0.7-x-vec-index folder rename. Idempotent; safe to call on
        # fresh trees (no-op) and on upgraded ones (rename + return).
        if paths.storage_dir.exists():
            migrate_qmd_corpus_to_corpus(paths.storage_dir)

        persist_dir = (
            Path(args.persist_dir) if args.persist_dir else paths.vec_index_dir
        )
        db_path = Path(args.db_path) if args.db_path else paths.graph_db

        summary = rebuild_vec_index(
            persist_dir=persist_dir,
            db_path=db_path,
            model_name=cfg.vec_index.model_name,
        )
        print(json.dumps(summary))
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"vec-index-rebuild failed: {exc}", file=sys.stderr)
        return 1


def rebuild_vec_index(
    *,
    persist_dir: Path,
    db_path: Path,
    model_name: str,
) -> dict:
    """Drop + repopulate the ``records`` collection from the SQLite
    ``records`` table.

    Drop-and-rebuild via :meth:`chromadb.Client.delete_collection` keeps the
    semantics clean (no stale ids carry across runs even if a record row
    was removed in the meantime).
    """
    import sqlite3

    started = time.monotonic()

    # Open once just to reach the underlying client, drop, then reopen so
    # the fresh collection has the right metadata (cosine space) applied.
    idx = RecordsVecIndex(persist_dir, model_name)
    try:
        idx._client.delete_collection("records")
    except Exception:
        # Collection didn't exist; ignore.
        pass
    idx = RecordsVecIndex(persist_dir, model_name)

    rows_scanned = 0
    empty_skipped = 0
    pending: list[tuple[str, str]] = []
    records_added = 0

    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        try:
            cur = conn.execute(
                "SELECT id, summary FROM records ORDER BY id"
            )
            for rid, summary in cur:
                rows_scanned += 1
                if not rid or not (summary or "").strip():
                    empty_skipped += 1
                    continue
                pending.append((rid, summary))
                if len(pending) >= _BATCH_SIZE:
                    idx.add_records_batch(pending)
                    records_added += len(pending)
                    pending = []
        finally:
            conn.close()

    if pending:
        idx.add_records_batch(pending)
        records_added += len(pending)

    elapsed = time.monotonic() - started
    return {
        "rows_scanned": rows_scanned,
        "records_added": records_added,
        "empty_skipped": empty_skipped,
        "elapsed_seconds": round(elapsed, 3),
    }
