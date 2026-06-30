"""SQLite connection factory for the active graph."""
from __future__ import annotations

import sqlite3
from pathlib import Path


def connect(db_path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # Wait up to 5s for a competing writer to release the lock before
    # raising ``OperationalError: database is locked``. With WAL mode
    # (set in schema.apply_migrations) actual contention is rare since
    # readers don't block writers, but if a writer is mid-transaction
    # this keeps the bridge from failing on a transient hold.
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn
