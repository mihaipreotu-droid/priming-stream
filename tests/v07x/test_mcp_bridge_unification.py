"""P1-A: MCP bridge unification — smoke tests for new A-pipeline paths.

Tests spec P1-A §6:
(a) dispatch graph_salient_context on a synthetic SQLite substrate:
    - output contains '### Semantic' and date labels;
    - a record with a rare term NOT touched by spread surfaces in
      '### Lexical' (regression Collins & Loftus).
(b) graph_disambiguate -> same two-bucket format.
(c) graph_spread -> items have source_date and kind; capped at max_records.
(d) existing tests are NOT weakened; this module only adds coverage.

Uses a real migrated tmp SQLite + a stub vec_index (no real fastembed /
ChromaDB). The stub follows the same interface contract as in
test_working_set.py (_StubVec) so walk_two_seeds runs end-to-end.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from priming_stream.core.db import connect
from priming_stream.core.graph_repo import GraphRepo
from priming_stream.core.models import Record
from priming_stream.core.schema import apply_migrations
from priming_stream.integrations.vec_index import VecHit
from priming_stream.mcp_server import server as server_mod


# ------------------------------------------------------------------ helpers


def _conn(tmp_path: Path):
    conn = connect(tmp_path / "graph.db")
    apply_migrations(conn)
    return conn


def _insert(
    repo: GraphRepo,
    rid: str,
    summary: str,
    *,
    kind: str = "claim",
    source_date: str | None = None,
    doc_key: str | None = None,
) -> None:
    repo.create_record(Record(
        id=rid,
        source_uri=f"owner://manual" if kind == "index_card" else f"cc://c/{rid}",
        anchor_offset_start=0,
        anchor_offset_end=0,
        summary=summary,
        created_at="2026-06-10T00:00:00Z",
        kind=kind,
        doc_key=doc_key,
        source_date=source_date,
    ))


# ------------------------------------------------------------------ stub vec_index


class _StubVec:
    """Scripted two-seed walk transport (single-seed / hop-0 only).

    ``prompt_hits``: canned hits returned for the prompt seed at hop 0.
    All other seeds and hop>0 entries return empty, terminating the walk
    at hop 0.
    """

    def __init__(self, prompt_hits: list[VecHit]) -> None:
        self._hits = prompt_hits

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        # Tag each text positionally: idx 0 = prompt seed.
        return [[float(i), 0.0] for i in range(len(texts))]

    def embeddings_for(self, record_ids: list[str]) -> dict[str, list[float]]:
        # Hop>0 sentinel: -1 → query_by_vecs returns [] → walk stops.
        return {rid: [-1.0, 0.0] for rid in record_ids}

    def query_by_vecs(
        self, vecs: list[list[float]], k: int
    ) -> list[list[VecHit]]:
        out: list[list[VecHit]] = []
        for v in vecs:
            idx = int(v[0])
            if idx == 0:
                out.append(list(self._hits))
            else:
                out.append([])
        return out

    def count(self) -> int:
        return 0


def _cfg(**overrides) -> Any:
    base = dict(
        decay=0.8,
        min_score=0.3,
        frontier_cap=10,
        k_per_query=10,
        max_hops=4,
        max_records=20,
        recency_strength=0.25,
        recency_age_span_days=180,
        recency_p_max=0.5,
        bucket_total=25,
        bucket_lexical=5,
        recency_filter_cutoff="",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ------------------------------------------------------------------ dispatch helper


def _dispatch(tool_name: str, args: dict, graph_db: Path) -> Any:
    """Call dispatch_tool with a normal (non-readonly) connection so tests
    can write records first and then dispatch read-only tools.

    dispatch_tool opens its OWN connection; we just use the regular helper.
    For tests we need to write before dispatching, so we write via a
    separate connection opened by the test fixture.
    """
    return server_mod.dispatch_tool(tool_name, args, graph_db)


def _dispatch_with_stub(
    tool_name: str,
    args: dict,
    graph_db: Path,
    stub_vec: _StubVec,
    bridge_cfg,
) -> Any:
    """Dispatch a tool but inject the stub vec_index and cfg via monkeypatch.

    Uses the handler directly (bypassing dispatch_tool's connection
    management) so the stub is used for the A-pipeline walk.
    """
    import sqlite3 as _sqlite3
    conn = _sqlite3.connect(str(graph_db))
    conn.row_factory = _sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    repo = GraphRepo(conn)

    from priming_stream.mcp_server import tools as tools_mod
    from unittest.mock import patch

    cfg_obj = SimpleNamespace(bridge=bridge_cfg)

    with (
        patch.object(tools_mod, "_get_vec_index", return_value=stub_vec),
        patch("priming_stream.mcp_server.tools.load_config", return_value=cfg_obj),
    ):
        handler = tools_mod.TOOLS[tool_name]
        result = handler(repo, args)
    conn.close()
    return result


# ================================================================== (a) graph_salient_context


def test_graph_salient_context_two_bucket_format(tmp_path):
    """graph_salient_context output contains ### Semantic and date labels."""
    db = tmp_path / "graph.db"
    conn = _conn(tmp_path)
    repo = GraphRepo(conn)
    _insert(repo, "rec_sem00001", "SemNet semantic network modelling architecture",
            source_date="2026-05-01T10:00:00Z")
    _insert(repo, "rec_sem00002", "bridge daemon warm latency budget",
            source_date="2026-04-15T08:00:00Z")
    conn.close()

    vec = _StubVec([
        VecHit(record_id="rec_sem00001", score=0.9, summary=""),
        VecHit(record_id="rec_sem00002", score=0.8, summary=""),
    ])
    cfg = _cfg()

    result = _dispatch_with_stub(
        "graph_salient_context",
        {"message": "ce arhitectura are bridge-ul SemNet?"},
        db,
        vec,
        cfg,
    )

    assert isinstance(result, str)
    assert "### Semantic" in result
    # Date labels should appear (A.5a)
    assert "2026-05-01" in result or "2026-04-15" in result


def test_graph_salient_context_collins_regression(tmp_path):
    """Collins & Loftus index_card surfaces in ### Lexical even when the
    dense walk misses it (regression: the rare term must NOT be missed)."""
    db = tmp_path / "graph.db"
    conn = _conn(tmp_path)
    repo = GraphRepo(conn)

    # The card the dense walk misses.
    _insert(
        repo, "rec_card0001",
        "Collins Loftus 1975 spreading activation theory of semantic memory",
        kind="index_card",
        doc_key="t:collins-loftus-1975",
        source_date="1975-01-01",
    )
    # Unrelated claims the dense walk surfaces.
    _insert(repo, "rec_claim001", "bridge daemon warm latency budget")
    _insert(repo, "rec_claim002", "sleep cycle consolidation of records")
    conn.close()

    # Dense walk returns ONLY the unrelated claims — the card is NOT activated.
    vec = _StubVec([
        VecHit(record_id="rec_claim001", score=0.9, summary=""),
        VecHit(record_id="rec_claim002", score=0.85, summary=""),
    ])
    cfg = _cfg()

    result = _dispatch_with_stub(
        "graph_salient_context",
        {"message": "ce zice collins loftus despre spreading activation?"},
        db,
        vec,
        cfg,
    )

    assert isinstance(result, str)
    # The card must appear somewhere in the output (lexical bucket picked it up).
    assert "rec_card0001" in result
    # And it should be in the Lexical section.
    assert "### Lexical" in result


def test_graph_salient_context_empty_message_returns_no_context(tmp_path):
    db = tmp_path / "graph.db"
    conn = _conn(tmp_path)
    repo = GraphRepo(conn)
    _insert(repo, "rec_x00001", "something")
    conn.close()

    vec = _StubVec([])
    result = _dispatch_with_stub(
        "graph_salient_context", {"message": ""}, db, vec, _cfg()
    )
    assert "(empty" in result.lower()


# ================================================================== (b) graph_disambiguate


def test_graph_disambiguate_two_bucket_format(tmp_path):
    """graph_disambiguate returns the same two-bucket markdown as graph_salient_context."""
    db = tmp_path / "graph.db"
    conn = _conn(tmp_path)
    repo = GraphRepo(conn)
    _insert(repo, "rec_dis00001", "PanelX n=76 measurement FMCG panel",
            source_date="2026-05-20T14:00:00Z")
    conn.close()

    vec = _StubVec([
        VecHit(record_id="rec_dis00001", score=0.9, summary=""),
    ])
    cfg = _cfg()

    result = _dispatch_with_stub(
        "graph_disambiguate",
        {"text": "the measurement from last month"},
        db,
        vec,
        cfg,
    )

    assert isinstance(result, str)
    assert "### Semantic" in result
    assert "rec_dis00001" in result
    # Date label should appear
    assert "2026-05-20" in result


def test_graph_disambiguate_collins_lexical_regression(tmp_path):
    """graph_disambiguate also picks up the Collins card via lexical bucket."""
    db = tmp_path / "graph.db"
    conn = _conn(tmp_path)
    repo = GraphRepo(conn)
    _insert(
        repo, "rec_card0002",
        "Collins Loftus spreading activation semantic memory theory",
        kind="index_card",
        source_date="1975-01-01",
    )
    _insert(repo, "rec_other001", "chromadb cosine distance handling")
    conn.close()

    vec = _StubVec([
        VecHit(record_id="rec_other001", score=0.9, summary=""),
    ])

    result = _dispatch_with_stub(
        "graph_disambiguate",
        {"text": "collins loftus semantic memory spreading"},
        db,
        vec,
        _cfg(),
    )

    assert isinstance(result, str)
    assert "rec_card0002" in result
    assert "### Lexical" in result


def test_graph_disambiguate_empty_text(tmp_path):
    db = tmp_path / "graph.db"
    conn = _conn(tmp_path)
    repo = GraphRepo(conn)
    _insert(repo, "rec_y00001", "something")
    conn.close()

    vec = _StubVec([])
    result = _dispatch_with_stub(
        "graph_disambiguate", {"text": ""}, db, vec, _cfg()
    )
    assert "(empty" in result.lower()


# ================================================================== (c) graph_spread


def test_graph_spread_items_have_source_date_and_kind(tmp_path):
    """graph_spread items carry source_date and kind (parity with search tools)."""
    db = tmp_path / "graph.db"
    conn = _conn(tmp_path)
    repo = GraphRepo(conn)
    _insert(repo, "rec_spr00001", "SemNet semantic network modelling",
            kind="claim", source_date="2026-05-10T00:00:00Z")
    _insert(repo, "rec_spr00002", "PanelX n=76 panel measurement",
            kind="index_card", source_date="2026-05-20T00:00:00Z")
    conn.close()

    vec = _StubVec([
        VecHit(record_id="rec_spr00001", score=0.9, summary=""),
        VecHit(record_id="rec_spr00002", score=0.8, summary=""),
    ])
    cfg = _cfg()

    result = _dispatch_with_stub(
        "graph_spread", {"text": "SemNet measurement panel"}, db, vec, cfg
    )

    assert isinstance(result, list)
    assert len(result) >= 1
    for item in result:
        assert "source_date" in item, f"missing source_date in {item}"
        assert "kind" in item, f"missing kind in {item}"
        assert "rank" in item
        assert "record_id" in item
        assert "summary" in item


def test_graph_spread_capped_at_max_records(tmp_path):
    """graph_spread respects max_records cap."""
    db = tmp_path / "graph.db"
    conn = _conn(tmp_path)
    repo = GraphRepo(conn)
    rids = [f"rec_{i:08x}" for i in range(10)]
    hits = []
    for rid in rids:
        _insert(repo, rid, f"summary {rid}", source_date="2026-01-01")
        hits.append(VecHit(record_id=rid, score=0.9, summary=""))
    conn.close()

    vec = _StubVec(hits)
    cfg = _cfg(max_records=3)

    result = _dispatch_with_stub(
        "graph_spread", {"text": "anything"}, db, vec, cfg
    )

    assert len(result) <= 3


def test_graph_spread_empty_text_returns_empty(tmp_path):
    db = tmp_path / "graph.db"
    conn = _conn(tmp_path)
    repo = GraphRepo(conn)
    _insert(repo, "rec_z00001", "something")
    conn.close()

    vec = _StubVec([])
    result = _dispatch_with_stub(
        "graph_spread", {"text": ""}, db, vec, _cfg()
    )
    assert result == []


def test_graph_spread_rank_is_contiguous(tmp_path):
    """Ranks in graph_spread output are 1-based and contiguous."""
    db = tmp_path / "graph.db"
    conn = _conn(tmp_path)
    repo = GraphRepo(conn)
    for i in range(3):
        _insert(repo, f"rec_r{i:07x}", f"summary {i}",
                source_date=f"2026-0{i+1}-01")
    conn.close()

    vec = _StubVec([
        VecHit(record_id=f"rec_r{i:07x}", score=0.9 - i * 0.05, summary="")
        for i in range(3)
    ])

    result = _dispatch_with_stub(
        "graph_spread", {"text": "summary"}, db, vec, _cfg()
    )

    ranks = [item["rank"] for item in result]
    assert ranks == list(range(1, len(result) + 1))
