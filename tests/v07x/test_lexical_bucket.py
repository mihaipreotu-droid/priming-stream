"""Bucket B lexical channel acceptance tests (Component A, A.2 / AC-lexical).

Builds a tmp SQLite on the real schema (``apply_migrations``) and seeds a
mix of an ``index_card`` naming a paper plus several ``claim`` records.
Exercises BM25 order, A-first dedup, no-threshold intake, kind-bias
(the Collins-citation behaviour), truncation, never-raises, and that
``ScoredRecord.record`` is a full Record (kind / source_date reachable).
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from priming_stream.bridge.lexical import _sanitize_for_fts5, lexical_bucket
from priming_stream.bridge.types import ScoredRecord
from priming_stream.core.db import connect
from priming_stream.core.models import Record, new_record_id, now_iso
from priming_stream.core.schema import apply_migrations


# -------------------------------------------------------------- fixtures


def _add_record(
    conn: sqlite3.Connection,
    summary: str,
    *,
    kind: str = "claim",
    source_date: str | None = None,
    doc_key: str | None = None,
    title: str | None = None,
    rec_id: str | None = None,
) -> str:
    """Insert one record via direct SQL so the FTS triggers fire."""
    rid = rec_id or new_record_id()
    conn.execute(
        "INSERT INTO records "
        "(id, source_uri, anchor_offset_start, anchor_offset_end, "
        "summary, created_at, source_date, kind, doc_key, source, "
        "content_hash, title, provisional) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            rid,
            "qmd://test/r.md" if kind == "claim" else "file:///doc.pdf",
            0,
            len(summary),
            summary,
            now_iso(),
            source_date,
            kind,
            doc_key,
            None,
            None,
            title,
            0,
        ),
    )
    conn.commit()
    return rid


def _make_db(tmp_path: Path) -> Path:
    db = tmp_path / "graph.db"
    conn = connect(db)
    try:
        apply_migrations(conn)
    finally:
        conn.close()
    return db


# ----------------------------------------------------------------- (a) order


def test_a_bm25_order_best_match_first(tmp_path):
    db = _make_db(tmp_path)
    conn = connect(db)
    try:
        # Two summaries contain 'bridge'; the one where it is the rarest /
        # most concentrated term should rank ahead under BM25.
        weak = _add_record(
            conn,
            "bridge across many other unrelated padding words here today",
        )
        strong = _add_record(conn, "bridge bridge")
        _add_record(conn, "totally unrelated content about pottery")

        out = lexical_bucket(
            conn, "bridge", limit=5, exclude_ids=set(), kind_bias=False
        )
    finally:
        conn.close()

    ids = [sr.record.id for sr in out]
    assert set(ids) == {weak, strong}
    # lower bm25 = better; results pre-sorted by query, so non-decreasing.
    assert out[0].score <= out[1].score
    assert ids[0] == strong


# ------------------------------------------------------------- (b) exclude


def test_b_exclude_ids_removed(tmp_path):
    db = _make_db(tmp_path)
    conn = connect(db)
    try:
        keep = _add_record(conn, "bridge spreading activation")
        drop = _add_record(conn, "bridge connection between records")

        out = lexical_bucket(
            conn, "bridge", limit=5, exclude_ids={drop}, kind_bias=False
        )
    finally:
        conn.close()

    ids = {sr.record.id for sr in out}
    assert drop not in ids
    assert keep in ids


# --------------------------------------------------------- (c) no threshold


def test_c_weak_rare_match_still_surfaces(tmp_path):
    db = _make_db(tmp_path)
    conn = connect(db)
    try:
        # One record holds a rare token buried in noise; it is the ONLY
        # match and must surface despite a weak BM25 score (no cutoff).
        weak = _add_record(
            conn,
            "a long padded summary mentioning marshmallow exactly once amid "
            "lots of other ordinary common filler words and phrases",
        )
        for i in range(5):
            _add_record(conn, f"unrelated common filler record number {i}")

        out = lexical_bucket(
            conn, "marshmallow", limit=5, exclude_ids=set(), kind_bias=False
        )
    finally:
        conn.close()

    assert [sr.record.id for sr in out] == [weak]


# ------------------------------------------------------------ (d) kind bias


def test_d_kind_bias_card_ahead_of_claims(tmp_path):
    db = _make_db(tmp_path)
    conn = connect(db)
    try:
        # The card names the paper; the claims discuss it too. With
        # kind_bias the card must lead even if a claim scores better BM25.
        card = _add_record(
            conn,
            "Collins Loftus 1975 spreading activation theory of semantic "
            "memory processing",
            kind="index_card",
            doc_key="t:collins-loftus-1975-spreading-activation",
            title="A Spreading-Activation Theory of Semantic Processing",
        )
        claim1 = _add_record(
            conn, "spreading activation activation activation semantic memory"
        )
        claim2 = _add_record(conn, "spreading activation in semantic networks")

        biased = lexical_bucket(
            conn, "spreading activation semantic", limit=5,
            exclude_ids=set(), kind_bias=True,
        )
        pure = lexical_bucket(
            conn, "spreading activation semantic", limit=5,
            exclude_ids=set(), kind_bias=False,
        )
    finally:
        conn.close()

    biased_ids = [sr.record.id for sr in biased]
    assert card in biased_ids
    assert biased_ids[0] == card  # card surfaced ahead of any claim

    pure_ids = [sr.record.id for sr in pure]
    assert set(pure_ids) == {card, claim1, claim2}
    # pure BM25: non-decreasing score, card NOT forced to front
    scores = [sr.score for sr in pure]
    assert scores == sorted(scores)


def test_d_kind_bias_preserves_bm25_within_kind(tmp_path):
    db = _make_db(tmp_path)
    conn = connect(db)
    try:
        _add_record(
            conn, "neural bridge card overview", kind="index_card",
            doc_key="t:bridge-card", title="Bridge Card",
        )
        claim_strong = _add_record(conn, "bridge bridge")
        claim_weak = _add_record(
            conn, "bridge among assorted unrelated padding words and phrases"
        )

        out = lexical_bucket(
            conn, "bridge", limit=5, exclude_ids=set(), kind_bias=True
        )
    finally:
        conn.close()

    ids = [sr.record.id for sr in out]
    # card first (bias), then claims in BM25 order (strong before weak).
    assert ids[0] != claim_strong and ids[0] != claim_weak
    assert ids.index(claim_strong) < ids.index(claim_weak)


# ----------------------------------------------------------- (e) truncation


def test_e_truncates_to_limit(tmp_path):
    db = _make_db(tmp_path)
    conn = connect(db)
    try:
        for i in range(8):
            _add_record(conn, f"bridge record variant number {i}")

        out = lexical_bucket(
            conn, "bridge", limit=3, exclude_ids=set(), kind_bias=False
        )
    finally:
        conn.close()

    assert len(out) == 3


# --------------------------------------------------------- (f) never raises


def test_f_missing_fts_table_returns_empty(tmp_path):
    # A DB with a bare records table but no records_fts virtual table.
    db = tmp_path / "bare.db"
    conn = sqlite3.connect(str(db))
    try:
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
            """
        )
        conn.execute(
            "INSERT INTO records (id, source_uri, anchor_offset_start, "
            "anchor_offset_end, summary, created_at) "
            "VALUES ('rec_x', 'qmd://t', 0, 5, 'bridge', '2026-01-01T00:00:00Z')"
        )
        conn.commit()
        out = lexical_bucket(
            conn, "bridge", limit=5, exclude_ids=set()
        )
    finally:
        conn.close()
    assert out == []


def test_f_empty_and_whitespace_prompt_returns_empty(tmp_path):
    db = _make_db(tmp_path)
    conn = connect(db)
    try:
        _add_record(conn, "bridge spreading activation")
        assert lexical_bucket(conn, "", limit=5, exclude_ids=set()) == []
        assert lexical_bucket(conn, "   ", limit=5, exclude_ids=set()) == []
        assert lexical_bucket(conn, "!! ??", limit=5, exclude_ids=set()) == []
    finally:
        conn.close()


def test_f_no_match_returns_empty(tmp_path):
    db = _make_db(tmp_path)
    conn = connect(db)
    try:
        _add_record(conn, "bridge spreading activation")
        out = lexical_bucket(
            conn, "zzzzzz_nonexistent_token", limit=5, exclude_ids=set()
        )
    finally:
        conn.close()
    assert out == []


def test_f_zero_or_negative_limit_returns_empty(tmp_path):
    db = _make_db(tmp_path)
    conn = connect(db)
    try:
        _add_record(conn, "bridge spreading activation")
        assert lexical_bucket(conn, "bridge", limit=0, exclude_ids=set()) == []
        assert lexical_bucket(conn, "bridge", limit=-1, exclude_ids=set()) == []
    finally:
        conn.close()


def test_f_nasty_query_does_not_raise(tmp_path):
    db = _make_db(tmp_path)
    conn = connect(db)
    try:
        _add_record(conn, "bridge alpha record")
        _add_record(conn, "decision outcome beta")
        nasty = [
            'bridge "quoted phrase" alpha',
            "alpha *wild*card*",
            "key: value: another:",
            "(parens) (more parens)",
            "NEAR(foo bar) baz",
            '"""""',
            "alpha AND OR NOT bridge",
        ]
        for q in nasty:
            out = lexical_bucket(conn, q, limit=5, exclude_ids=set())
            assert isinstance(out, list)
    finally:
        conn.close()


# --------------------------------------------- (g) full Record reachable


def test_g_returns_scored_record_with_full_record(tmp_path):
    db = _make_db(tmp_path)
    conn = connect(db)
    try:
        dated = _add_record(
            conn, "bridge dated claim record", source_date="2026-05-01T10:40:00Z"
        )
        _add_record(
            conn, "bridge card record", kind="index_card",
            doc_key="t:bridge-doc", title="Bridge Doc",
        )

        out = lexical_bucket(conn, "bridge", limit=5, exclude_ids=set())
    finally:
        conn.close()

    assert out and all(isinstance(sr, ScoredRecord) for sr in out)
    for sr in out:
        # full Record: kind + source_date are reachable attributes.
        assert sr.record.kind in ("claim", "index_card")
        _ = sr.record.source_date  # accessible (None for the card)
        assert isinstance(sr.score, float)
    by_id = {sr.record.id: sr for sr in out}
    assert by_id[dated].record.source_date == "2026-05-01T10:40:00Z"


# ------------------------------------------ (h) OR semantics (partial match)


def test_h_partial_match_sentence_surfaces_record(tmp_path):
    """A multi-token natural sentence whose tokens only PARTIALLY appear in a
    target summary still surfaces it (A.2 OR semantics). This is the test that
    would FAIL under the old implicit-AND join — most prompt tokens are absent
    from the summary, so AND requires-every-token would match zero rows."""
    db = _make_db(tmp_path)
    conn = connect(db)
    try:
        target = _add_record(
            conn,
            "Collins Loftus 1975 spreading activation theory of semantic memory",
            kind="index_card",
            doc_key="t:collins-loftus-1975",
            title="A Spreading-Activation Theory of Semantic Processing",
        )
        _add_record(conn, "totally unrelated content about pottery glazing")

        prompt = "ce zice collins & loftus si cum am folosit asta la meshgraph?"
        out = lexical_bucket(conn, prompt, limit=5, exclude_ids=set())

        # OR: the bare 'collins'/'loftus' overlap surfaces the card.
        ids = {sr.record.id for sr in out}
        assert target in ids

        # Prove the AND counterfactual: every prompt token required → zero.
        and_match = " ".join(f'"{t}"' for t in re.findall(r"\w+", prompt)
                             if len(t) >= 2)
        and_rows = conn.execute(
            "SELECT r.id FROM records r "
            "JOIN records_fts f ON r.rowid = f.rowid "
            "WHERE f.summary MATCH ?",
            (and_match,),
        ).fetchall()
        assert and_rows == []
    finally:
        conn.close()


# ----------------------------------------------------------- sanitizer unit


def test_sanitizer_quotes_and_drops_single_chars():
    # A.2: tokens OR-joined (down-rank, not exclude), each double-quoted,
    # single-char tokens dropped as BM25 noise.
    assert _sanitize_for_fts5("foo bar baz") == '"foo" OR "bar" OR "baz"'
    assert _sanitize_for_fts5("a b c hello world") == '"hello" OR "world"'


def test_sanitizer_empty_for_tokenless():
    assert _sanitize_for_fts5("") == ""
    assert _sanitize_for_fts5("   ") == ""
    assert _sanitize_for_fts5("!!! ??? :::") == ""


def test_sanitizer_unicode_tokens():
    assert (
        _sanitize_for_fts5("raționament holistic")
        == '"raționament" OR "holistic"'
    )
