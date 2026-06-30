"""Stdlib lexical fallback — FTS5 BM25 over ``records.summary``.

Used by the hot-path hook when the daemon is cold or unreachable. Sub-50ms
on the canonical 155-record DB; opens the SQLite file read-only so the
hook never accidentally mutates the graph.

The query string is sanitized for FTS5's ``MATCH`` syntax: tokens are
extracted with a Unicode word regex and each is wrapped in double quotes
so special characters (``"``, ``*``, ``:``, parentheses, NEAR) don't
parse-error the underlying virtual table.

NEVER raises — returns ``[]`` on missing DB, empty query, or any error.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _sanitize_for_fts5(query_text: str) -> str:
    """Tokenize + quote each token; join with implicit-AND whitespace.

    Single-char tokens are dropped (BM25 noise; the default FTS5 tokenizer
    already filters most). Returns empty string for empty input — callers
    skip the SQL entirely in that case.
    """
    tokens = _TOKEN_RE.findall(query_text or "")
    tokens = [t for t in tokens if len(t) >= 2]
    if not tokens:
        return ""
    return " ".join(f'"{t}"' for t in tokens)


def search(db_path: Path, query_text: str, k: int = 10) -> list[tuple[str, str]]:
    """Return up to ``k`` ``(record_id, summary)`` BM25-ranked matches.

    Empty list on empty query, missing DB, no matches, or any error.
    """
    try:
        match = _sanitize_for_fts5(query_text)
        if not match:
            return []
        if not Path(db_path).exists():
            return []
        uri = f"file:{Path(db_path).as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            cur = conn.execute(
                "SELECT r.id, r.summary FROM records r "
                "JOIN records_fts f ON r.rowid = f.rowid "
                "WHERE f.summary MATCH ? "
                "ORDER BY bm25(records_fts) "
                "LIMIT ?",
                (match, int(k)),
            )
            return [(row[0], row[1]) for row in cur.fetchall()]
        finally:
            conn.close()
    except Exception:
        return []
