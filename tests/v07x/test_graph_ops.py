"""v0.7-x-vec-index W-F: graph_ops unit tests with stub :class:`RecordsVecIndex`.

Each function in ``Priming Stream.graph_ops`` is exercised against a real
``GraphRepo`` (migrated tmp SQLite) and a stub vec_index that returns
canned :class:`VecHit` lists. No real fastembed / ChromaDB.

``graph_search_chunks`` is dropped in v0.7-x-vec-index — no tests for it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from priming_stream.core.db import connect
from priming_stream.core.graph_repo import GraphRepo
from priming_stream.core.models import Record, now_iso
from priming_stream.core.schema import apply_migrations
from priming_stream.graph_ops import (
    graph_chunk_around_anchor,
    graph_records,
    graph_search_records,
    graph_spread_op,
    graph_stats,
)
from priming_stream.integrations.vec_index import VecHit


# -- helpers --------------------------------------------------------------


def _repo(tmp_path: Path) -> GraphRepo:
    conn = connect(tmp_path / "graph.db")
    apply_migrations(conn)
    return GraphRepo(conn)


def _make_record(repo: GraphRepo, rid: str, summary: str,
                 anchor_start=None, anchor_end=None,
                 source_uri: str | None = None) -> Record:
    uri = source_uri or f"qmd://priming-stream-records/{rid}.md"
    r = Record(
        id=rid,
        source_uri=uri,
        anchor_offset_start=anchor_start,
        anchor_offset_end=anchor_end,
        summary=summary,
        created_at=now_iso(),
    )
    repo.create_record(r)
    return r


@dataclass
class _StubVecIndex:
    """Fixed-return stub.

    ``hits`` is the canned list returned by ``search`` (hop-0 text queries)
    and by ``query_by_vecs`` for the first seed. Implements the A-pipeline
    interface (``embed_texts`` / ``embeddings_for`` / ``query_by_vecs``) so
    ``walk_two_seeds`` works without real fastembed/ChromaDB.

    Hop>0 lookups (``embeddings_for``) return sentinel vectors that
    ``query_by_vecs`` maps to empty results, terminating the walk at hop 0.
    """
    hits: list[VecHit] = field(default_factory=list)
    count_value: int = 0
    raise_on_count: bool = False

    def search(self, query_text, k):
        _ = query_text, k
        return list(self.hits)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        # Return tagged sentinel vectors: index 0 = first seed (prompt).
        return [[float(i), 0.0] for i in range(len(texts))]

    def embeddings_for(self, record_ids: list[str]) -> dict[str, list[float]]:
        # Hop>0: sentinel -1 → query_by_vecs returns empty → walk stops.
        return {rid: [-1.0, 0.0] for rid in record_ids}

    def query_by_vecs(
        self, vecs: list[list[float]], k: int
    ) -> list[list[VecHit]]:
        out: list[list[VecHit]] = []
        for v in vecs:
            idx = int(v[0])
            # idx==0 → first seed (prompt) → return canned hits.
            # idx<0 or out-of-range → hop>0 sentinel → no hits.
            if idx == 0:
                out.append(list(self.hits))
            else:
                out.append([])
        return out

    def count(self) -> int:
        if self.raise_on_count:
            raise RuntimeError("vec_index down")
        return self.count_value


def _cfg_bridge():
    from types import SimpleNamespace
    return SimpleNamespace(
        decay=0.8, min_score=0.3, frontier_cap=10,
        k_per_query=10, max_hops=4, max_records=20,
    )


# -- graph_search_records -------------------------------------------------


def test_search_records_returns_enriched_hits(tmp_path):
    repo = _repo(tmp_path)
    _make_record(repo, "rec_00000001", "alpha summary")
    _make_record(repo, "rec_00000002", "beta summary")
    vec = _StubVecIndex(hits=[
        VecHit(record_id="rec_00000001", score=0.9, summary="alpha summary"),
        VecHit(record_id="rec_00000002", score=0.7, summary="beta summary"),
    ])
    out = graph_search_records("anything", 10, vec, repo)
    assert len(out) == 2
    assert out[0]["record_id"] == "rec_00000001"
    assert out[0]["summary"] == "alpha summary"
    assert out[0]["score"] == 0.9
    assert "qmd://priming-stream-records/rec_00000001.md" in out[0]["source_uri"]


def test_search_records_skips_unknown_ids(tmp_path):
    repo = _repo(tmp_path)
    _make_record(repo, "rec_00000001", "alpha")
    vec = _StubVecIndex(hits=[
        VecHit(record_id="rec_00000001", score=0.9, summary="alpha"),
        VecHit(record_id="rec_deadbeef", score=0.5, summary="ghost"),
    ])
    out = graph_search_records("x", 10, vec, repo)
    assert [h["record_id"] for h in out] == ["rec_00000001"]


def test_search_records_empty_query_returns_empty(tmp_path):
    repo = _repo(tmp_path)
    vec = _StubVecIndex(hits=[])
    assert graph_search_records("", 10, vec, repo) == []


# -- graph_records --------------------------------------------------------


def test_graph_records_returns_full_dict(tmp_path):
    repo = _repo(tmp_path)
    _make_record(
        repo, "rec_00000001", "alpha", anchor_start=10, anchor_end=42,
    )
    out = graph_records("rec_00000001", repo)
    assert out is not None
    assert out["id"] == "rec_00000001"
    assert out["summary"] == "alpha"
    assert out["anchor_offset_start"] == 10
    assert out["anchor_offset_end"] == 42
    assert out["source_uri"].startswith("qmd://")
    assert out["created_at"]  # non-empty


def test_graph_records_unknown_returns_none(tmp_path):
    repo = _repo(tmp_path)
    assert graph_records("rec_00000001", repo) is None
    assert graph_records("", repo) is None


# -- graph_spread_op ------------------------------------------------------


def test_spread_op_returns_ranked_dicts(tmp_path):
    """graph_spread_op returns rank-ordered dicts with source_date and kind.

    Adapted from the legacy spread() test: now uses walk_two_seeds via
    the full A-pipeline stub (_StubVecIndex with embed_texts/query_by_vecs).
    """
    repo = _repo(tmp_path)
    _make_record(repo, "rec_00000001", "alpha")
    _make_record(repo, "rec_00000002", "beta")
    vec = _StubVecIndex(hits=[
        VecHit(record_id="rec_00000001", score=0.9, summary="alpha"),
        VecHit(record_id="rec_00000002", score=0.6, summary="beta"),
    ])
    out = graph_spread_op("hello", vec, repo, _cfg_bridge())
    assert [(d["rank"], d["record_id"]) for d in out] == [
        (1, "rec_00000001"), (2, "rec_00000002"),
    ]
    assert out[0]["summary"] == "alpha"
    # New: source_date and kind fields must be present (parity with search tools).
    assert "source_date" in out[0]
    assert "kind" in out[0]
    assert out[0]["kind"] == "claim"  # default kind for _make_record


def test_spread_op_empty_text_returns_empty(tmp_path):
    repo = _repo(tmp_path)
    vec = _StubVecIndex(hits=[])
    assert graph_spread_op("", vec, repo, _cfg_bridge()) == []


# -- graph_stats ----------------------------------------------------------


def test_stats_returns_records_and_vec_index_size(tmp_path):
    repo = _repo(tmp_path)
    _make_record(repo, "rec_00000001", "alpha")
    _make_record(repo, "rec_00000002", "beta")
    cycle_id = repo.start_sleep_cycle(started_at=now_iso())
    repo.finish_sleep_cycle(
        cycle_id, completed_at=now_iso(),
        chunks_materialized=2, records_created=2, records_skipped=0,
        metrics_json="{}", notes=None,
    )
    vec = _StubVecIndex(count_value=2)
    out = graph_stats(repo, vec)
    assert out["records_count"] == 2
    assert out["last_sleep_cycle"] is not None
    assert out["last_sleep_cycle"]["chunks_materialized"] == 2
    assert out["vec_index_size"] == 2
    # qmd_collections field must be gone.
    assert "qmd_collections" not in out


def test_stats_no_sleep_cycle_is_none(tmp_path):
    repo = _repo(tmp_path)
    vec = _StubVecIndex(count_value=0)
    out = graph_stats(repo, vec)
    assert out["records_count"] == 0
    assert out["last_sleep_cycle"] is None
    assert out["vec_index_size"] == 0


def test_stats_vec_index_error_degrades_to_none(tmp_path):
    repo = _repo(tmp_path)
    vec = _StubVecIndex(raise_on_count=True)
    out = graph_stats(repo, vec)
    assert out["records_count"] == 0
    assert out["vec_index_size"] is None
