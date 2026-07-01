"""Acceptance + unit tests for the bridge orchestrator (v0.7-x Component A).

``build_priming`` composes the four frozen leaf modules into the read-time
A-pipeline. These tests exercise it against a REAL tmp SQLite (so FTS5
triggers populate ``records_fts`` and ``get_record`` works) plus a STUB
vec_index that returns canned ``VecHit``s for the two-seed walk — no real
fastembed / ChromaDB.

The three named fixtures (Collins & Loftus, bimodal, PanelX) are FROZEN
acceptance criteria for the two-seed bridge walk.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from priming_stream.bridge.working_set import build_priming
from priming_stream.core.db import connect
from priming_stream.core.graph_repo import GraphRepo
from priming_stream.core.models import Record
from priming_stream.core.schema import apply_migrations
from priming_stream.integrations.vec_index import VecHit


# -- config + db helpers --------------------------------------------------


def _cfg(**overrides):
    """Duck-typed BridgeConfig stand-in (frozen Component-A defaults)."""
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


def _conn(tmp_path):
    conn = connect(tmp_path / "graph.db")
    apply_migrations(conn)
    return conn


def _insert(repo: GraphRepo, rid: str, summary: str, *,
            kind: str = "claim", source_date: str | None = None,
            doc_key: str | None = None) -> None:
    repo.create_record(Record(
        id=rid,
        source_uri="owner://manual" if kind == "index_card" else f"qmd://c/{rid}.md",
        anchor_offset_start=0,
        anchor_offset_end=0,
        summary=summary,
        created_at="2026-06-01T00:00:00Z",
        kind=kind,
        doc_key=doc_key,
        source_date=source_date,
    ))


# -- stub vec_index -------------------------------------------------------


class _StubVec:
    """Scripted two-seed walk transport.

    ``seed_hits`` maps a seed lineage label ('prompt' | 'response') to the
    canned VecHit list returned for that seed at hop 0. Hop>0 queries (driven
    by ``embeddings_for`` + ``query_by_vecs`` on stored source vectors) return
    empty, so the walk terminates at hop 0 — enough for these acceptance
    assertions, which turn on seed-level activation.

    The seed-batch order is what ``walk_two_seeds`` builds: only non-empty
    seeds, in the fixed order ('prompt', 'response'). We tag each embedded seed
    vector with its lineage so ``query_by_vecs`` can route by tag.
    """

    def __init__(self, seed_hits: dict[str, list[VecHit]]) -> None:
        self._seed_hits = seed_hits
        # Set when embed_texts is called with the live ordered seed list.
        self._seed_order: list[str] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        # walk_two_seeds embeds the non-empty seeds in order (prompt, response).
        # We can't see the lineage labels here, only the texts — but the caller
        # passes them in the same order it builds the frontier, so we hand back
        # tagged sentinel vectors positionally. The lineage is recovered in
        # query_by_vecs via the sentinel's first element (an index into the
        # ordered non-empty seed list this stub was told about at construction).
        self._seed_order = [lbl for lbl in ("prompt", "response")
                            if lbl in self._present]
        return [[float(i), 0.0] for i in range(len(texts))]

    def embeddings_for(self, record_ids: list[str]) -> dict[str, list[float]]:
        # Hop>0 source vectors: hand back a sentinel that query_by_vecs maps to
        # "no further hits", terminating the walk after hop 0.
        return {rid: [-1.0, 0.0] for rid in record_ids}

    def query_by_vecs(self, vecs: list[list[float]], k: int) -> list[list[VecHit]]:
        out: list[list[VecHit]] = []
        for v in vecs:
            idx = int(v[0])
            if idx < 0 or idx >= len(self._seed_order):
                out.append([])  # hop>0 sentinel → walk stops
                continue
            lineage = self._seed_order[idx]
            out.append(list(self._seed_hits.get(lineage, [])))
        return out

    # set by the fixtures so embed_texts knows which lineages are present
    _present: set[str] = {"prompt"}


def _vec(seed_hits: dict[str, list[VecHit]], present=("prompt",)) -> _StubVec:
    s = _StubVec(seed_hits)
    s._present = set(present)
    return s


def _hit(rid: str, score: float) -> VecHit:
    return VecHit(record_id=rid, score=score, summary="")


_NOW = datetime(2026, 6, 4, 12, 0, 0, tzinfo=timezone.utc)


# -- AC-citation (Collins & Loftus) --------------------------------------


def test_ac_citation_collins_loftus_card_surfaces_in_lexical(tmp_path):
    """A real sentence NAMING the paper surfaces its index_card in bucket B
    even though dense spreading misses it (it would surface in ZERO buckets
    today). The prompt is a natural question — most of its tokens are absent
    from the card summary — so under the OLD implicit-AND join it returns zero
    (proven inline below). OR surfaces it on the bare 'collins'/'loftus' hit."""
    conn = _conn(tmp_path)
    repo = GraphRepo(conn)
    _insert(repo, "rec_card0001",
            "Collins Loftus 1975 spreading activation theory of semantic "
            "memory processing", kind="index_card",
            doc_key="t:collins-1975-spreading-activation")
    # Unrelated claims that the dense walk DOES surface.
    _insert(repo, "rec_claim001", "the bridge daemon warm latency budget")
    _insert(repo, "rec_claim002", "sleep cycle consolidation of records")
    _insert(repo, "rec_claim003", "chromadb cosine space distance handling")

    # Stub: the seed surfaces only the unrelated claims — the card is NOT
    # semantically retrieved.
    vec = _vec({"prompt": [
        _hit("rec_claim001", 0.9),
        _hit("rec_claim002", 0.85),
        _hit("rec_claim003", 0.8),
    ]})

    # The real source case: a natural-language question naming the paper. Its
    # tokens (ce/zice/collins/loftus/si/cum/am/folosit/asta/la/meshgraph) are NOT a
    # subset of the card summary, so implicit-AND would MATCH zero rows.
    prompt = "ce zice collins & loftus si cum am folosit asta la meshgraph?"
    result = build_priming(
        prompt, "",
        vec_index=vec, repo=repo, conn=conn, cfg=_cfg(), now=_NOW,
    )

    semantic_ids = {sr.record.id for sr in result.semantic}
    lexical_ids = {sr.record.id for sr in result.lexical}
    # Card NOT in semantic (the dense walk missed it) but IS in lexical.
    assert "rec_card0001" not in semantic_ids
    assert "rec_card0001" in lexical_ids

    # Load-bearing proof: the SAME prompt under the OLD implicit-AND join
    # (whitespace, not OR) returns zero — so the OR fix is what surfaces it.
    import re as _re
    toks = [t for t in _re.findall(r"\w+", prompt) if len(t) >= 2]
    and_match = " ".join(f'"{t}"' for t in toks)
    and_rows = conn.execute(
        "SELECT r.id FROM records r "
        "JOIN records_fts f ON r.rowid = f.rowid "
        "WHERE f.summary MATCH ?",
        (and_match,),
    ).fetchall()
    assert and_rows == []  # implicit-AND: every prompt token required → zero


# -- AC-bimodal (demografie + Meshgraph) ---------------------------------------


def test_ac_bimodal_topic_in_semantic_demografie_in_lexical(tmp_path):
    """A natural disjoint-theme sentence surfaces BOTH themes — Meshgraph via the
    semantic spread, demografie via the lexical citation channel; demografie
    must NOT leak into semantic. The prompt is a real question whose tokens
    are NOT a subset of any one summary, so under the OLD implicit-AND join
    bucket B returns zero (proven inline); OR surfaces the demografie record."""
    conn = _conn(tmp_path)
    repo = GraphRepo(conn)
    _insert(repo, "rec_msh00001", "Meshgraph knowledge-graph modelling cognitive structure")
    _insert(repo, "rec_msh00002", "Meshgraph models entity relationship graphs")
    _insert(repo, "rec_msh00003", "knowledge-graph modelling research notes")
    _insert(repo, "rec_demo0001",
            "demografie segmentare populatie pe varste si venituri")

    # Stub: the prompt surfaces only the Meshgraph claims semantically (bucket A) —
    # the stub is independent of lexical token overlap.
    vec = _vec({"prompt": [
        _hit("rec_msh00001", 0.9),
        _hit("rec_msh00002", 0.88),
        _hit("rec_msh00003", 0.85),
    ]})

    # One natural sentence spanning both themes. Its tokens overlap the
    # demografie summary only PARTIALLY (demografie/pe), so OR surfaces that
    # record in bucket B; the stub routes the SAME prompt to the Meshgraph claims
    # semantically. Bimodal: one prompt → two disjoint themes.
    prompt = "cum se leaga metodologia Meshgraph de segmentarea pe demografie?"
    result = build_priming(
        prompt, "",
        vec_index=vec, repo=repo, conn=conn, cfg=_cfg(), now=_NOW,
    )

    semantic_ids = {sr.record.id for sr in result.semantic}
    lexical_ids = {sr.record.id for sr in result.lexical}
    assert {"rec_msh00001", "rec_msh00002", "rec_msh00003"} <= semantic_ids
    assert "rec_demo0001" in lexical_ids
    assert "rec_demo0001" not in semantic_ids

    # Load-bearing proof: the SAME prompt under the OLD implicit-AND join
    # matches zero rows (no summary contains EVERY prompt token).
    import re as _re
    toks = [t for t in _re.findall(r"\w+", prompt) if len(t) >= 2]
    and_match = " ".join(f'"{t}"' for t in toks)
    and_rows = conn.execute(
        "SELECT r.id FROM records r "
        "JOIN records_fts f ON r.rowid = f.rowid "
        "WHERE f.summary MATCH ?",
        (and_match,),
    ).fetchall()
    assert and_rows == []


# -- AC-panelx (A.5a surface + gentle recency) ----------------------------


def test_ac_panelx_both_surface_newer_ranks_at_least_as_high(tmp_path):
    """The two PanelX measurements both surface in bucket A; with default
    gentle strength the newer (n=76, 2026-05-01) ranks >= the older
    (n=65, 2026-04-09). Equal raw activation, so recency is the tiebreaker."""
    conn = _conn(tmp_path)
    repo = GraphRepo(conn)
    _insert(repo, "rec_panelx065", "PanelX n=65 measurement",
            source_date="2026-04-09T10:00:00Z")
    _insert(repo, "rec_panelx076", "PanelX n=76 measurement",
            source_date="2026-05-01T10:40:00Z")

    # Near-equal raw activation from the seed.
    vec = _vec({"prompt": [
        _hit("rec_panelx065", 0.80),
        _hit("rec_panelx076", 0.80),
    ]})

    result = build_priming(
        "what is the PanelX sample size",
        "",
        vec_index=vec, repo=repo, conn=conn, cfg=_cfg(), now=_NOW,
    )

    sem_ids = [sr.record.id for sr in result.semantic]
    assert "rec_panelx065" in sem_ids
    assert "rec_panelx076" in sem_ids
    # Newer ranks at least as high (lower index) as older.
    assert sem_ids.index("rec_panelx076") <= sem_ids.index("rec_panelx065")
    # And strictly higher recency-weighted score (newer is less penalized).
    by_id = {sr.record.id: sr.score for sr in result.semantic}
    assert by_id["rec_panelx076"] > by_id["rec_panelx065"]


def test_ac_panelx_item_builder_yields_source_date(tmp_path):
    """A.5a plumbing: the daemon item-builder carries source_date through,
    so render can show the date inline."""
    from priming_stream.daemon.server import _scored_to_item

    conn = _conn(tmp_path)
    repo = GraphRepo(conn)
    _insert(repo, "rec_panelx076", "PanelX n=76 measurement",
            source_date="2026-05-01T10:40:00Z")
    vec = _vec({"prompt": [_hit("rec_panelx076", 0.8)]})

    result = build_priming(
        "PanelX sample", "",
        vec_index=vec, repo=repo, conn=conn, cfg=_cfg(), now=_NOW,
    )
    item = _scored_to_item(result.semantic[0], 1)
    assert item["source_date"] == "2026-05-01T10:40:00Z"
    assert item["kind"] == "claim"
    assert item["rank"] == 1


# -- build_priming units --------------------------------------------------


def test_a_first_dedup_record_in_both_appears_only_in_semantic(tmp_path):
    """A record that both the dense walk AND the lexical FTS5 surface appears
    only in bucket A (A-first dedup — anti-redundancy)."""
    conn = _conn(tmp_path)
    repo = GraphRepo(conn)
    _insert(repo, "rec_shared01", "embedding latency budget overlap term")
    _insert(repo, "rec_only_lex", "embedding latency budget unrelated tail")

    # Dense walk surfaces ONLY rec_shared01; FTS5 over the prompt matches BOTH
    # (they share 'embedding latency budget').
    vec = _vec({"prompt": [_hit("rec_shared01", 0.9)]})

    result = build_priming(
        "embedding latency budget", "",
        vec_index=vec, repo=repo, conn=conn, cfg=_cfg(), now=_NOW,
    )
    sem_ids = {sr.record.id for sr in result.semantic}
    lex_ids = {sr.record.id for sr in result.lexical}
    assert "rec_shared01" in sem_ids
    # Deduped out of lexical because it's already in semantic.
    assert "rec_shared01" not in lex_ids


def test_empty_prompt_and_prev_yields_empty_semantic(tmp_path):
    conn = _conn(tmp_path)
    repo = GraphRepo(conn)
    _insert(repo, "rec_x0000001", "some record")
    vec = _vec({"prompt": [_hit("rec_x0000001", 0.9)]})

    result = build_priming(
        "", "", vec_index=vec, repo=repo, conn=conn, cfg=_cfg(), now=_NOW,
    )
    assert result.semantic == []
    # Empty prompt → no FTS5 match either.
    assert result.lexical == []


def test_now_defaults_to_utc_when_omitted(tmp_path):
    """Omitting ``now`` must not raise; defaults to current UTC."""
    conn = _conn(tmp_path)
    repo = GraphRepo(conn)
    _insert(repo, "rec_dated001", "dated claim",
            source_date="2026-05-01T10:00:00Z")
    vec = _vec({"prompt": [_hit("rec_dated001", 0.8)]})

    result = build_priming(
        "dated claim", "", vec_index=vec, repo=repo, conn=conn, cfg=_cfg(),
    )
    assert {sr.record.id for sr in result.semantic} == {"rec_dated001"}


def test_semantic_budget_truncates_to_bucket_total_minus_lexical(tmp_path):
    """Bucket A is truncated to ``bucket_total - bucket_lexical`` (20)."""
    conn = _conn(tmp_path)
    repo = GraphRepo(conn)
    ids = [f"rec_{i:08x}" for i in range(30)]
    hits = []
    for i, rid in enumerate(ids):
        _insert(repo, rid, f"summary alpha {rid}")
        hits.append(_hit(rid, 0.9 - i * 0.005))
    vec = _vec({"prompt": hits})

    result = build_priming(
        "alpha", "", vec_index=vec, repo=repo, conn=conn,
        cfg=_cfg(k_per_query=30), now=_NOW,
    )
    # bucket_total(25) - bucket_lexical(5) = 20.
    assert len(result.semantic) == 20
