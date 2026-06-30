"""Lexical fallback acceptance tests (spec §D4-D5).

Builds a small migrated DB with hand-crafted summaries, then queries via
``fallback_lexical.search``. Verifies BM25 ranking, empty / no-match
handling, and FTS5-special-token sanitization.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from priming_stream.core.db import connect
from priming_stream.core.models import Record, new_record_id, now_iso
from priming_stream.core.schema import apply_migrations
from priming_stream.daemon.fallback_lexical import _sanitize_for_fts5, search


# -------------------------------------------------------------- fixtures


def _seed_records(conn: sqlite3.Connection, summaries: list[str]) -> list[str]:
    ids = []
    for s in summaries:
        rec = Record(
            id=new_record_id(),
            source_uri="qmd://test/r.md",
            anchor_offset_start=0,
            anchor_offset_end=len(s),
            summary=s,
            created_at=now_iso(),
        )
        conn.execute(
            "INSERT INTO records "
            "(id, source_uri, anchor_offset_start, anchor_offset_end, "
            "summary, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (rec.id, rec.source_uri, rec.anchor_offset_start,
             rec.anchor_offset_end, rec.summary, rec.created_at),
        )
        ids.append(rec.id)
    conn.commit()
    return ids


def _make_db(tmp_path: Path, summaries: list[str]) -> Path:
    db = tmp_path / "graph.db"
    conn = connect(db)
    try:
        apply_migrations(conn)
        _seed_records(conn, summaries)
    finally:
        conn.close()
    return db


# ---------------------------------------------------------------- D4 tests


def test_d4_search_returns_matches(tmp_path):
    db = _make_db(tmp_path, [
        "bridge spreading activation in the Priming Stream",
        "totally unrelated content about pottery",
        "another bridge connection between two records",
        "lone keyword content marshmallow",
        "the third bridge document expounds at length",
    ])
    hits = search(db, "bridge", k=10)
    assert len(hits) == 3  # three summaries contain 'bridge'
    for rid, summary in hits:
        assert "bridge" in summary
        assert rid.startswith("rec_")


def test_d4_search_respects_k(tmp_path):
    db = _make_db(tmp_path, [
        "alpha bridge one",
        "beta bridge two",
        "gamma bridge three",
        "delta bridge four",
        "epsilon bridge five",
    ])
    hits = search(db, "bridge", k=2)
    assert len(hits) == 2


def test_d4_empty_query_returns_empty(tmp_path):
    db = _make_db(tmp_path, ["alpha bridge"])
    assert search(db, "", k=10) == []
    assert search(db, "   ", k=10) == []
    assert search(db, "!!", k=10) == []  # no word tokens


def test_d4_no_match_returns_empty(tmp_path):
    db = _make_db(tmp_path, ["alpha bridge"])
    assert search(db, "zzzzzz_nonexistent", k=10) == []


def test_d4_missing_db_returns_empty(tmp_path):
    missing = tmp_path / "does-not-exist.db"
    assert search(missing, "bridge", k=10) == []


def test_d4_corrupt_db_returns_empty(tmp_path):
    """Anything that fails to open as SQLite must not raise."""
    bad = tmp_path / "garbage.db"
    bad.write_text("not a sqlite db at all", encoding="utf-8")
    assert search(bad, "bridge", k=10) == []


# ---------------------------------------------------------------- D5 tests


def test_d5_sanitizer_quotes_tokens():
    out = _sanitize_for_fts5("foo bar baz")
    assert out == '"foo" "bar" "baz"'


def test_d5_sanitizer_drops_single_chars():
    out = _sanitize_for_fts5("a b c hello world")
    assert out == '"hello" "world"'


def test_d5_sanitizer_empty_input():
    assert _sanitize_for_fts5("") == ""
    assert _sanitize_for_fts5("   ") == ""
    assert _sanitize_for_fts5("!!! ??? :::") == ""


def test_d5_search_tolerates_special_chars(tmp_path):
    """Quotes / asterisks / colons / parens must not crash FTS5 parsing."""
    db = _make_db(tmp_path, [
        "bridge alpha record",
        "decision outcome beta",
    ])
    nasty_queries = [
        'bridge "quoted phrase" alpha',
        "alpha *wild*card*",
        "key: value: another:",
        "(parens) (more parens)",
        'NEAR(foo bar) baz',
        '"""""',
        "alpha AND OR NOT bridge",
    ]
    for q in nasty_queries:
        # Must not raise; return list (possibly empty).
        hits = search(db, q, k=10)
        assert isinstance(hits, list)


def test_d5_unicode_query_tokenized(tmp_path):
    db = _make_db(tmp_path, [
        "raționament holistic peste substrat",
    ])
    # Romanian diacritics tokenize as word chars under \w+ unicode flag.
    hits = search(db, "raționament", k=10)
    assert len(hits) == 1
    assert "raționament" in hits[0][1]
