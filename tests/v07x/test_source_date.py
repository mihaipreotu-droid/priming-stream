"""Tests for v0.7-x-B source_date derivation (core/source_date.py)."""
from __future__ import annotations

from pathlib import Path

from priming_stream.core import source_date as sd


_EXPORT = (
    "---\n"
    "chunk_id: export_sid_p0\n"
    "started_at: 2026-02-01T08:00:00.000000Z\n"
    "ended_at: 2026-02-01T09:00:00.000000Z\n"
    "---\n"
    "\n"
    "## User — 2026-02-01T08:00:05.000000Z\n"
    "primul mesaj cu diacritice: șțăî\n"
    "\n"
    "## Assistant — 2026-02-01T08:01:30.000000Z\n"
    "al doilea turn, mai jos în fișier\n"
)


def _make_corpus(tmp_path: Path, export_text: str = _EXPORT) -> tuple[Path, Path, str]:
    """Build a minimal corpus with one export. Returns (storage, corpus, uri)."""
    corpus = tmp_path / "corpus"
    sid_dir = corpus / "imports" / "claude_ai_export" / "sid"
    sid_dir.mkdir(parents=True)
    (sid_dir / "export_sid_p0.md").write_text(export_text, encoding="utf-8")
    uri = "qmd://priming-stream-imports/claude_ai_export/sid/export_sid_p0.md"
    return tmp_path, corpus, uri


def test_started_at_and_turn_offsets():
    assert sd.started_at_of(_EXPORT) == "2026-02-01T08:00:00.000000Z"
    turns = sd.turn_offsets(_EXPORT)
    assert [ts for _, ts in turns] == [
        "2026-02-01T08:00:05.000000Z",
        "2026-02-01T08:01:30.000000Z",
    ]
    # offsets are char positions, ascending
    assert turns[0][0] < turns[1][0]


def test_nearest_turn_ts_at_or_before():
    turns = sd.turn_offsets(_EXPORT)
    first_off, second_off = turns[0][0], turns[1][0]
    # before any header -> None (caller falls back to started_at)
    assert sd.nearest_turn_ts(turns, first_off - 1) is None
    # exactly on / after first header but before second -> first ts
    assert sd.nearest_turn_ts(turns, first_off) == turns[0][1]
    assert sd.nearest_turn_ts(turns, second_off - 1) == turns[0][1]
    # at/after second -> second ts
    assert sd.nearest_turn_ts(turns, second_off + 5) == turns[1][1]


def test_resolve_source_date_picks_anchored_turn(tmp_path):
    storage, corpus, uri = _make_corpus(tmp_path)
    turns = sd.turn_offsets(_EXPORT)
    got = sd.resolve_source_date(
        uri, turns[1][0] + 2, storage_dir=storage, corpus_dir=corpus,
    )
    assert got == "2026-02-01T08:01:30.000000Z"


def test_resolve_source_date_falls_back_to_started_at(tmp_path):
    storage, corpus, uri = _make_corpus(tmp_path)
    # anchor before the first turn header -> no turn matches -> started_at
    got = sd.resolve_source_date(uri, 0, storage_dir=storage, corpus_dir=corpus)
    assert got == "2026-02-01T08:00:00.000000Z"


def test_resolve_source_date_none_for_non_qmd(tmp_path):
    storage, corpus, _ = _make_corpus(tmp_path)
    assert sd.resolve_source_date(
        "file:///C:/x/y.md", 0, storage_dir=storage, corpus_dir=corpus,
    ) is None
    assert sd.resolve_source_date(
        "doc://url:example.org/p", 0, storage_dir=storage, corpus_dir=corpus,
    ) is None


def test_resolve_source_date_none_for_missing_export(tmp_path):
    storage, corpus, _ = _make_corpus(tmp_path)
    missing = "qmd://priming-stream-imports/claude_ai_export/sid/export_nope_p0.md"
    assert sd.resolve_source_date(
        missing, 0, storage_dir=storage, corpus_dir=corpus,
    ) is None


def test_ensure_source_dates_in_db(tmp_path):
    from priming_stream.core.db import connect
    from priming_stream.core.graph_repo import GraphRepo
    from priming_stream.core.models import Record
    from priming_stream.core.schema import apply_migrations

    storage, corpus, uri = _make_corpus(tmp_path)
    turns = sd.turn_offsets(_EXPORT)

    conn = connect(tmp_path / "graph.db")
    apply_migrations(conn)
    repo = GraphRepo(conn)
    # a staged conversation claim -> gets source_date
    repo.stage_record(Record(
        id="rec_a", source_uri=uri,
        anchor_offset_start=turns[1][0] + 2, anchor_offset_end=999,
        summary="claim body", created_at="2026-06-02T00:00:00Z",
    ))
    # a staged index_card -> no conversation date, left untouched
    repo.stage_record(Record(
        id="rec_b", source_uri="doc://url:example.org/p",
        anchor_offset_start=0, anchor_offset_end=0,
        summary="card body", created_at="2026-06-02T00:00:00Z",
        kind="index_card", doc_key="url:example.org/p", provisional=True,
    ))

    m = sd.ensure_source_dates_in_db(
        conn, storage_dir=storage, corpus_dir=corpus,
    )
    assert m["written"] == 1
    assert m["no_date"] == 1
    assert repo.get_staged("rec_a").source_date == "2026-02-01T08:01:30.000000Z"
    assert repo.get_staged("rec_b").source_date is None

    # second run is idempotent
    m2 = sd.ensure_source_dates_in_db(
        conn, storage_dir=storage, corpus_dir=corpus,
    )
    assert m2["written"] == 0
    assert m2["skipped_existing"] == 1
    conn.close()


def test_ensure_source_dates_in_db_does_not_touch_promoted_rows(tmp_path):
    """The derivation is scoped to STAGING — live ``records`` rows (already
    backfilled) are never rewritten."""
    from priming_stream.core.db import connect
    from priming_stream.core.graph_repo import GraphRepo
    from priming_stream.core.models import Record
    from priming_stream.core.schema import apply_migrations

    storage, corpus, uri = _make_corpus(tmp_path)
    conn = connect(tmp_path / "graph.db")
    apply_migrations(conn)
    repo = GraphRepo(conn)
    repo.create_record(Record(
        id="rec_live", source_uri=uri,
        anchor_offset_start=0, anchor_offset_end=1,
        summary="live claim", created_at="2026-06-02T00:00:00Z",
        source_date=None,
    ))
    m = sd.ensure_source_dates_in_db(
        conn, storage_dir=storage, corpus_dir=corpus,
    )
    assert m == {"written": 0, "skipped_existing": 0, "no_date": 0, "malformed": 0}
    assert repo.get_record("rec_live").source_date is None
    conn.close()
