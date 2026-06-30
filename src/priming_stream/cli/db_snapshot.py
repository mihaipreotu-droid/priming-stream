"""``prime db-snapshot`` — versioned, WAL-safe backups of ``graph.db``.

SQL-canonical (2026-06-12): the SQLite database IS the substrate's source
of truth (the ``.md``-per-record layer was retired), so its rebuild-safety
story is **snapshot history**, not a parallel store. This command produces
a consistent point-in-time copy via the SQLite backup API — safe while the
daemon or a sleep cycle holds the database open (WAL readers/writer are
unaffected; the backup sees a consistent transaction boundary).

Snapshots land in ``storage/_db_snapshots/graph-<UTC timestamp>.db``.
``storage/`` is gitignored by design (personal data, publication-bound
repo), so durable history = this snapshot dir (+ optionally a private git
repo over ``storage/`` if the owner ever wants per-snapshot diffs).
Restore = stop the daemon, copy a snapshot over ``storage/graph.db``,
``prime vec-index-rebuild``, restart the daemon.

ChromaDB needs no snapshot of its own — it re-derives from the records
table (``vec-index-rebuild``).
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from priming_stream.core.config import load_config
from priming_stream.core.paths import resolve_paths

SNAPSHOT_DIRNAME = "_db_snapshots"


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "db-snapshot",
        help="write a WAL-safe point-in-time backup of graph.db to "
             "storage/_db_snapshots/ (the substrate's rebuild source)",
    )
    p.add_argument(
        "--out", default=None,
        help="explicit output file path (default: "
             "storage/_db_snapshots/graph-<UTC ts>.db)",
    )
    p.set_defaults(func=cmd_db_snapshot)


def snapshot_db(db_path: Path, out_path: Path) -> dict:
    """Backup ``db_path`` → ``out_path`` via the SQLite backup API.
    Returns ``{path, records, size_bytes}``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(str(db_path))
    try:
        dst = sqlite3.connect(str(out_path))
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()

    check = sqlite3.connect(str(out_path))
    try:
        records = check.execute("SELECT count(*) FROM records").fetchone()[0]
    finally:
        check.close()
    return {
        "path": str(out_path),
        "records": records,
        "size_bytes": out_path.stat().st_size,
    }


def cmd_db_snapshot(args: argparse.Namespace) -> int:
    try:
        cfg = load_config()
        paths = resolve_paths(cfg, project_root=Path.cwd())
        if not paths.graph_db.exists():
            print(
                f"no graph database at {paths.graph_db} — run 'prime init' first",
                file=sys.stderr,
            )
            return 1

        if args.out:
            out_path = Path(args.out)
        else:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            out_path = (
                paths.storage_dir / SNAPSHOT_DIRNAME / f"graph-{ts}.db"
            )

        info = snapshot_db(paths.graph_db, out_path)
        print(
            f"db-snapshot: {info['records']} records -> {info['path']} "
            f"({info['size_bytes'] / 1024 / 1024:.1f} MB)"
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"db-snapshot failed: {exc}", file=sys.stderr)
        return 1
