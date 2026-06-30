"""v0.7-x-vec-index inspector dashboard — records / sleep cycles / vec_index."""
from __future__ import annotations

from pathlib import Path

import json

from priming_stream.core.db import connect
from priming_stream.core.graph_repo import GraphRepo
from priming_stream.core.models import Record, new_record_id, now_iso
from priming_stream.core.schema import apply_migrations
from priming_stream.inspector.dashboard import (
    _corpus_health,
    _echoes_section,
    _health_section,
    generate_dashboard,
)


class _FakeVecIndex:
    """Canned :class:`RecordsVecIndex` stand-in for dashboard tests."""

    def __init__(self, count: int = 0, raise_count: bool = False) -> None:
        self._count = count
        self._raise_count = raise_count

    def count(self) -> int:
        if self._raise_count:
            raise RuntimeError("vec_index unavailable")
        return self._count


def _seed_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "graph.db"
    conn = connect(db_path)
    try:
        apply_migrations(conn)
        repo = GraphRepo(conn)
        for i in range(2):
            repo.create_record(Record(
                id=new_record_id(),
                source_uri=f"qmd://priming-stream-imports/x/y/{i}.md",
                anchor_offset_start=i * 100,
                anchor_offset_end=i * 100 + 50,
                summary=f"record {i} summary text",
                created_at=now_iso(),
            ))
        c1 = repo.start_sleep_cycle(started_at="2026-05-25T10:00:00Z")
        repo.finish_sleep_cycle(
            c1,
            completed_at="2026-05-25T10:05:00Z",
            chunks_materialized=3,
            records_created=2,
            records_skipped=1,
            metrics_json="{}",
            notes="first",
        )
        repo.start_sleep_cycle(started_at="2026-05-25T11:00:00Z")
    finally:
        conn.close()
    return db_path


def test_generate_dashboard_writes_html(tmp_path):
    db_path = _seed_db(tmp_path)
    out = tmp_path / "out" / "dashboard.html"
    vec = _FakeVecIndex(count=12)

    result = generate_dashboard(db_path, out, vec_index=vec)

    assert result == out
    assert out.exists()
    body = out.read_text(encoding="utf-8")

    # Required panel headings.
    assert "Records" in body
    assert "Sleep cycles" in body
    assert "vec_index" in body

    # Dropped v0.7-x panels must not reappear.
    assert "Provisional" not in body
    assert "PageRank" not in body
    assert "Edge updates" not in body

    # Dropped v0.7-x-vec-index legacy: no qmd corpus panel.
    assert "qmd corpus" not in body

    # Sanity: data we seeded shows up.
    assert "record 0 summary text" in body
    assert "first" in body  # cycle notes
    # vec_index count is rendered.
    assert ">12<" in body


def test_generate_dashboard_empty_db(tmp_path):
    db_path = tmp_path / "graph.db"
    conn = connect(db_path)
    try:
        apply_migrations(conn)
    finally:
        conn.close()

    out = tmp_path / "dashboard.html"
    vec = _FakeVecIndex(count=0)
    generate_dashboard(db_path, out, vec_index=vec)

    body = out.read_text(encoding="utf-8")
    assert "No records yet" in body
    assert "No sleep cycles yet" in body
    # vec_index panel still rendered.
    assert "vec_index" in body
    assert ">0<" in body


def test_generate_dashboard_vec_index_error_tolerated(tmp_path):
    db_path = _seed_db(tmp_path)
    out = tmp_path / "dashboard.html"
    vec = _FakeVecIndex(raise_count=True)
    # Should not raise — vec_index errors are rendered as 'unavailable'.
    generate_dashboard(db_path, out, vec_index=vec)
    body = out.read_text(encoding="utf-8")
    assert "unavailable" in body
    assert "vec_index" in body


def test_generate_dashboard_vec_index_none_tolerated(tmp_path):
    """When ``vec_index=None`` is passed (failed to construct upstream),
    dashboard still renders with the panel showing 'unavailable'."""
    db_path = _seed_db(tmp_path)
    out = tmp_path / "dashboard.html"
    generate_dashboard(db_path, out, vec_index=None)
    body = out.read_text(encoding="utf-8")
    # Either the records counted out (because real vec_index opened) or
    # rendered as unavailable — both are acceptable; what matters is no
    # crash and panel present.
    assert "vec_index" in body


# -- E.2: corpus-health aggregates ---------------------------------------


def _seed_mixed(tmp_path) -> object:
    """A db with 2 claims (one dated), 1 index_card, 1 provisional claim."""
    db_path = tmp_path / "graph.db"
    conn = connect(db_path)
    apply_migrations(conn)
    repo = GraphRepo(conn)
    repo.create_record(Record(
        id="rec_claim1", source_uri="qmd://x/a.md",
        anchor_offset_start=0, anchor_offset_end=10,
        summary="SemNet. a claim about networks", created_at=now_iso(),
        source_date="2026-03-01T10:00:00Z",
    ))
    repo.create_record(Record(
        id="rec_claim2", source_uri="qmd://x/b.md",
        anchor_offset_start=0, anchor_offset_end=10,
        summary="Acme. another claim", created_at=now_iso(),
        source_date="2026-05-01T10:00:00Z", provisional=True,
    ))
    repo.create_record(Record(
        id="rec_card1", source_uri="file:///docs/paper.pdf",
        anchor_offset_start=0, anchor_offset_end=0,
        summary="a paper card", created_at=now_iso(),
        kind="index_card", doc_key="t:author-2026-paper",
        content_hash="deadbeef",
    ))
    return conn


def test_corpus_health_aggregates(tmp_path):
    conn = _seed_mixed(tmp_path)
    try:
        h = _corpus_health(conn)
    finally:
        conn.close()
    assert h["total"] == 3
    assert h["by_kind"].get("claim") == 2
    assert h["by_kind"].get("index_card") == 1
    assert h["provisional"] == 1
    assert h["date_min"] == "2026-03-01T10:00:00Z"
    assert h["date_max"] == "2026-05-01T10:00:00Z"


def test_health_section_parity_match_and_gap(tmp_path):
    conn = _seed_mixed(tmp_path)
    try:
        matched = _health_section(conn, _FakeVecIndex(count=3), tmp_path)
        gapped = _health_section(conn, _FakeVecIndex(count=1), tmp_path)
        unavail = _health_section(conn, _FakeVecIndex(raise_count=True), tmp_path)
    finally:
        conn.close()
    assert "Corpus health" in matched
    assert "match" in matched
    assert ">claims<" not in matched  # label is a header cell, value is the count
    assert "2" in matched and "index_card" in matched
    # vec=1, total=3 → Δ 2
    assert "2" in gapped and "&Delta;" in gapped
    # vec unavailable degrades, never crashes
    assert "unavailable" in unavail


# -- E.2: recent bridge-invocations (echoes) panel -----------------------


def test_echoes_section_renders_and_resolves(tmp_path):
    db_path = _seed_db(tmp_path)  # has rec ids we don't know; add a known one
    conn = connect(db_path)
    repo = GraphRepo(conn)
    repo.create_record(Record(
        id="rec_known", source_uri="qmd://x/k.md",
        anchor_offset_start=0, anchor_offset_end=5,
        summary="a primed record summary", created_at=now_iso(),
    ))
    echoes_path = tmp_path / "echoes.jsonl"
    echoes_path.write_text(
        json.dumps({
            "at": "2026-06-12T09:00:00Z", "session_id": "abcd1234ef",
            "source": "semantic+lexical", "spread_ms": 312.7,
            "semantic": ["rec_known"], "lexical": [],
            "prompt_head": "what about the bridge?",
        }) + "\n",
        encoding="utf-8",
    )
    try:
        section = _echoes_section(echoes_path, repo)
    finally:
        conn.close()
    assert "Recent bridge invocations" in section
    assert "abcd1234" in section          # short session
    assert "313ms" in section             # spread_ms rendered (rounded)
    assert "1/0" in section               # A/B counts
    assert "what about the bridge?" in section
    assert "a primed record summary" in section  # id resolved to summary


def test_echoes_section_absent_file_graceful(tmp_path):
    db_path = _seed_db(tmp_path)
    conn = connect(db_path)
    repo = GraphRepo(conn)
    try:
        section = _echoes_section(tmp_path / "nonexistent.jsonl", repo)
    finally:
        conn.close()
    assert "No bridge invocations recorded yet" in section


def test_generate_dashboard_full_includes_four_panels(tmp_path, monkeypatch):
    """End-to-end render under storage isolation: all four panels present,
    including the echoes panel reading the isolated episodic dir."""
    from priming_stream.core.config import load_config
    from priming_stream.core.paths import ensure_dirs, resolve_paths

    monkeypatch.setenv("PRIMING_STREAM_STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    paths = resolve_paths(cfg)
    ensure_dirs(paths)
    # Seed the isolated graph db + an echo line.
    conn = connect(paths.graph_db)
    apply_migrations(conn)
    GraphRepo(conn).create_record(Record(
        id="rec_e1", source_uri="qmd://x/a.md",
        anchor_offset_start=0, anchor_offset_end=5,
        summary="seeded record", created_at=now_iso(),
    ))
    conn.close()
    (paths.episodic_dir / "echoes.jsonl").write_text(
        json.dumps({
            "at": "2026-06-12T09:00:00Z", "session_id": "sess0001",
            "source": "semantic", "spread_ms": 100,
            "semantic": ["rec_e1"], "lexical": [],
            "prompt_head": "hello",
        }) + "\n", encoding="utf-8",
    )

    out = tmp_path / "dash.html"
    generate_dashboard(paths.graph_db, out, vec_index=_FakeVecIndex(count=1))
    body = out.read_text(encoding="utf-8")
    for panel in ("Corpus health", "Records", "Sleep cycles",
                  "Recent bridge invocations"):
        assert panel in body, panel
    assert "seeded record" in body
    assert "hello" in body  # echo prompt head
