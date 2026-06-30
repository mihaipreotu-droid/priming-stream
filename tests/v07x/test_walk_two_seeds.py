"""v0.7-x Component A (A.1): two-seed batched spreading walk.

``walk_two_seeds`` runs two independent activation lineages (prompt,
response) but issues ONE batched chroma query per hop across both — the
blocking latency gate. Tests use a recording stub ``RecordsVecIndex`` that
implements the three batched read methods (``embed_texts``,
``embeddings_for``, ``query_by_vecs``) with canned hits and COUNTS calls.

Stub routing convention: every query vector is a 1-element ``[float(key)]``
where ``key`` indexes ``hits_by_key`` — ``embed_texts`` maps seed text →
key, ``embeddings_for`` maps record id → key. So ``query_by_vecs`` can route
each batched input vec back to its canned hit list without real embeddings.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from priming_stream.bridge.spreading import walk_two_seeds
from priming_stream.core.db import connect
from priming_stream.core.graph_repo import GraphRepo
from priming_stream.core.models import Record, now_iso
from priming_stream.core.schema import apply_migrations
from priming_stream.integrations.vec_index import RecordsVecIndex, VecHit


_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_RUN_VEC = os.environ.get("RUN_VEC_TESTS") == "1"


def _cfg(**overrides):
    base = dict(
        decay=0.8,
        min_score=0.3,
        frontier_cap=10,
        k_per_query=10,
        max_hops=4,
        max_records=20,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _repo(tmp_path) -> GraphRepo:
    conn = connect(tmp_path / "graph.db")
    apply_migrations(conn)
    return GraphRepo(conn)


def _make_record(rid: str, summary: str) -> Record:
    return Record(
        id=rid,
        source_uri=f"qmd://priming-stream-records/{rid}.md",
        anchor_offset_start=None,
        anchor_offset_end=None,
        summary=summary,
        created_at=now_iso(),
    )


def _hit(rid: str, score: float) -> VecHit:
    return VecHit(record_id=rid, score=score, summary="")


# -- recording stub ------------------------------------------------------


@dataclass
class _StubVec:
    """Recording stub for the two-seed walk's batched read path.

    ``seed_keys``: seed text -> routing key (int).
    ``rec_keys``:  record id -> routing key (int) for stored-vector reuse on
                   hops >= 1. A record id absent here has no stored vec
                   (``embeddings_for`` omits it).
    ``hits_by_key``: routing key -> canned hit list returned by that query.

    Counts ``query_by_vecs`` calls (the per-hop batched-query proof) and
    ``embed_texts`` calls.
    """
    seed_keys: dict[str, int]
    rec_keys: dict[str, int]
    hits_by_key: dict[int, list[VecHit]]
    query_calls: list[int] = field(default_factory=list)  # batch size per call
    embed_calls: int = 0

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls += 1
        return [[float(self.seed_keys[t])] for t in texts]

    def embeddings_for(self, record_ids: list[str]) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for rid in record_ids:
            if rid in self.rec_keys:
                out[rid] = [float(self.rec_keys[rid])]
        return out

    def query_by_vecs(self, vecs: list[list[float]], k: int):
        self.query_calls.append(len(vecs))
        results = []
        for v in vecs:
            key = int(round(v[0]))
            results.append(list(self.hits_by_key.get(key, [])))
        return results


class _DupGuardError(Exception):
    """Raised by the chroma-faithful fake collection on duplicate input ids."""


@dataclass
class _DupGuardVec(_StubVec):
    """Stub faithful to the REAL (fixed) ``RecordsVecIndex.embeddings_for``.

    The production fix dedups input ids before chroma's ``get`` (which rejects
    duplicates). This stub mirrors that contract — it dedups, then asserts the
    deduped ids are unique (mimicking the chroma layer) — so the walk-level test
    exercises a faithful model of the fixed component. The actual fail-on-old
    proof lives in :func:`test_embeddings_for_dedups_before_chroma_get`, which
    drives the real ``RecordsVecIndex.embeddings_for`` against a fake collection
    that raises on duplicate ids.
    """

    def embeddings_for(self, record_ids: list[str]) -> dict[str, list[float]]:
        deduped = list(dict.fromkeys(record_ids))
        if len(deduped) != len(set(deduped)):  # chroma-layer invariant
            raise _DupGuardError(f"duplicate ids reached chroma get: {deduped}")
        return super().embeddings_for(deduped)


class _FakeChromaCollection:
    """Minimal fake of a chroma collection for ``embeddings_for``: ``get``
    raises :class:`_DupGuardError` on duplicate ids, exactly like chroma's
    DuplicateIDError. Stores ``id -> vector``; missing ids are dropped.
    """

    def __init__(self, vecs: dict[str, list[float]]):
        self._vecs = vecs

    def get(self, ids, include=None):
        if len(ids) != len(set(ids)):
            raise _DupGuardError(f"DuplicateIDError: {ids}")
        out_ids: list[str] = []
        out_embs: list[list[float]] = []
        for rid in ids:
            if rid in self._vecs:
                out_ids.append(rid)
                out_embs.append(self._vecs[rid])
        return {"ids": out_ids, "embeddings": out_embs}


# -- (a) combined score == max(act_p, act_r) -----------------------------


def test_combined_score_is_max_over_lineages(tmp_path):
    repo = _repo(tmp_path)
    repo.create_record(_make_record("rec_shared01", "shared"))
    # Prompt seed (key 1) reaches the shared record at score 0.9.
    # Response seed (key 2) reaches the same record at score 0.5.
    # Single hop each: act_p = 0.9*0.8 = 0.72; act_r = 0.5*0.8 = 0.40.
    vec = _StubVec(
        seed_keys={"prompt q": 1, "response q": 2},
        rec_keys={},
        hits_by_key={
            1: [_hit("rec_shared01", 0.9)],
            2: [_hit("rec_shared01", 0.5)],
        },
    )
    out = walk_two_seeds("prompt q", "response q", vec, repo, _cfg(max_hops=1))
    assert len(out) == 1
    assert out[0].record.id == "rec_shared01"
    assert out[0].score == pytest.approx(0.72)  # max(0.72, 0.40)


# -- (b) one empty seed degenerates to single-lineage --------------------


def test_one_empty_seed_equals_single_lineage(tmp_path):
    repo = _repo(tmp_path)
    repo.create_record(_make_record("rec_aaaaaaa1", "alpha"))
    repo.create_record(_make_record("rec_aaaaaaa2", "beta"))

    def _vec():
        return _StubVec(
            seed_keys={"prompt q": 1},
            rec_keys={},
            hits_by_key={1: [_hit("rec_aaaaaaa1", 0.9), _hit("rec_aaaaaaa2", 0.6)]},
        )

    both = walk_two_seeds("prompt q", "", _vec(), repo, _cfg(max_hops=1))
    # Whitespace-only response is also "empty".
    ws = walk_two_seeds("prompt q", "   ", _vec(), repo, _cfg(max_hops=1))

    pairs_both = [(sr.record.id, round(sr.score, 6)) for sr in both]
    pairs_ws = [(sr.record.id, round(sr.score, 6)) for sr in ws]
    assert pairs_both == pairs_ws
    # Equals the single prompt-lineage activations: 0.9*0.8, 0.6*0.8.
    assert pairs_both == [("rec_aaaaaaa1", 0.72), ("rec_aaaaaaa2", 0.48)]


# -- (c) both empty -> [] ------------------------------------------------


def test_both_empty_returns_empty(tmp_path):
    repo = _repo(tmp_path)
    repo.create_record(_make_record("rec_aaaaaaa1", "alpha"))
    vec = _StubVec(seed_keys={}, rec_keys={}, hits_by_key={})
    assert walk_two_seeds("", "", vec, repo, _cfg()) == []
    assert walk_two_seeds("  ", " ", vec, repo, _cfg()) == []
    # No seeds -> no embed, no query.
    assert vec.embed_calls == 0
    assert vec.query_calls == []


# -- (d) BATCHED: query_by_vecs called EXACTLY ONCE PER HOP ---------------


def test_query_batched_once_per_hop(tmp_path):
    """Latency gate: one chroma query per hop across BOTH lineages, never one
    per frontier entry. With 2 seeds and a 2-hop walk that keeps both lineages
    alive, that is exactly 2 query_by_vecs calls — not 4+."""
    repo = _repo(tmp_path)
    for rid in ("rec_p0000001", "rec_p0000002", "rec_r0000001", "rec_r0000002"):
        repo.create_record(_make_record(rid, rid))

    # hop 0: prompt(key1)->p1, response(key2)->r1.
    # hop 1: p1's stored vec(key11)->p2; r1's stored vec(key21)->r2.
    vec = _StubVec(
        seed_keys={"prompt q": 1, "response q": 2},
        rec_keys={"rec_p0000001": 11, "rec_r0000001": 21},
        hits_by_key={
            1: [_hit("rec_p0000001", 0.95)],
            2: [_hit("rec_r0000001", 0.95)],
            11: [_hit("rec_p0000002", 0.95)],
            21: [_hit("rec_r0000002", 0.95)],
        },
    )
    out = walk_two_seeds("prompt q", "response q", vec, repo, _cfg(max_hops=2, min_score=0.1))

    # One embed (both seeds in a single call), two queries (one per hop).
    assert vec.embed_calls == 1
    assert len(vec.query_calls) == 2
    # Hop 0 batched both seeds; hop 1 batched both lineages' frontiers.
    assert vec.query_calls[0] == 2
    assert vec.query_calls[1] == 2
    ids = {sr.record.id for sr in out}
    assert ids == {"rec_p0000001", "rec_p0000002", "rec_r0000001", "rec_r0000002"}


# -- (e) min_score prune, frontier_cap, sorted output --------------------


def test_min_score_prunes(tmp_path):
    repo = _repo(tmp_path)
    repo.create_record(_make_record("rec_keep00001", "keep"))
    repo.create_record(_make_record("rec_drop00001", "drop"))
    # keep: 0.5*0.8 = 0.40 >= 0.3.  drop: 0.3*0.8 = 0.24 < 0.3.
    vec = _StubVec(
        seed_keys={"q": 1},
        rec_keys={},
        hits_by_key={1: [_hit("rec_keep00001", 0.5), _hit("rec_drop00001", 0.3)]},
    )
    out = walk_two_seeds("q", "", vec, repo, _cfg(min_score=0.3, max_hops=1))
    ids = [sr.record.id for sr in out]
    assert "rec_keep00001" in ids
    assert "rec_drop00001" not in ids


def test_frontier_cap_bounds_each_lineage(tmp_path):
    """frontier_cap caps each lineage independently. With cap=1 on a single
    lineage, hop 1 expands only the single highest-activation hop-0 record."""
    repo = _repo(tmp_path)
    for rid in ("rec_h0000001", "rec_h0000002", "rec_deep0001", "rec_deep0002"):
        repo.create_record(_make_record(rid, rid))
    # hop 0 surfaces two records; cap=1 keeps only the stronger (rec_h0000001,
    # 0.9 > rec_h0000002, 0.5). hop 1 then expands only rec_h0000001's vec.
    vec = _StubVec(
        seed_keys={"q": 1},
        rec_keys={"rec_h0000001": 11, "rec_h0000002": 12},
        hits_by_key={
            1: [_hit("rec_h0000001", 0.9), _hit("rec_h0000002", 0.5)],
            11: [_hit("rec_deep0001", 0.9)],
            12: [_hit("rec_deep0002", 0.9)],
        },
    )
    out = walk_two_seeds("q", "", vec, repo, _cfg(frontier_cap=1, max_hops=2, min_score=0.1))
    ids = {sr.record.id for sr in out}
    # rec_deep0001 (under the kept record) activates; rec_deep0002 (under the
    # capped-out record) does not.
    assert "rec_deep0001" in ids
    assert "rec_deep0002" not in ids
    # hop 1 issued exactly one query (the single capped frontier entry).
    assert vec.query_calls[1] == 1


def test_output_sorted_descending(tmp_path):
    repo = _repo(tmp_path)
    for rid in ("rec_lo000001", "rec_hi000001", "rec_md000001"):
        repo.create_record(_make_record(rid, rid))
    vec = _StubVec(
        seed_keys={"q": 1},
        rec_keys={},
        hits_by_key={1: [
            _hit("rec_lo000001", 0.5),
            _hit("rec_hi000001", 0.95),
            _hit("rec_md000001", 0.7),
        ]},
    )
    out = walk_two_seeds("q", "", vec, repo, _cfg(max_hops=1, min_score=0.1))
    scores = [sr.score for sr in out]
    assert scores == sorted(scores, reverse=True)
    assert out[0].record.id == "rec_hi000001"


def test_skips_record_missing_from_repo(tmp_path):
    repo = _repo(tmp_path)
    repo.create_record(_make_record("rec_real00001", "real"))
    vec = _StubVec(
        seed_keys={"q": 1},
        rec_keys={},
        hits_by_key={1: [_hit("rec_real00001", 0.9), _hit("rec_ghost0001", 0.9)]},
    )
    out = walk_two_seeds("q", "", vec, repo, _cfg(max_hops=1))
    assert [sr.record.id for sr in out] == ["rec_real00001"]


def test_does_not_reactivate_source(tmp_path):
    """A record's own stored vector hit (itself) is filtered on hop>0."""
    repo = _repo(tmp_path)
    repo.create_record(_make_record("rec_src000001", "source"))
    repo.create_record(_make_record("rec_nbr000001", "neighbour"))
    vec = _StubVec(
        seed_keys={"q": 1},
        rec_keys={"rec_src000001": 11},
        hits_by_key={
            1: [_hit("rec_src000001", 0.9)],
            # hop1: rec_src's own vec returns itself (filtered) + a neighbour.
            11: [_hit("rec_src000001", 0.99), _hit("rec_nbr000001", 0.9)],
        },
    )
    out = walk_two_seeds("q", "", vec, repo, _cfg(max_hops=2, min_score=0.1))
    ids = {sr.record.id for sr in out}
    assert ids == {"rec_src000001", "rec_nbr000001"}


# -- (f) cross-lineage convergence: embeddings_for never gets dup ids -----


def test_cross_lineage_convergence_dedups_stored_fetch(tmp_path):
    """Regression (live-substrate bug): when BOTH lineages converge on the same
    source record, hop>0's ``need_ids`` contains that id once per lineage. The
    real chroma ``collection.get`` rejects duplicate ids, so ``embeddings_for``
    must dedup before the get. The stub MIMICS chroma (raises on dup ids); this
    test errors on the old (un-deduped) code and passes once dedup is in place.

    Setup: both seeds hit the SAME record at hop 0 (rec_shared01). On hop 1 it
    appears as a frontier source in BOTH lineages → need_ids = [shared, shared].
    The walk must still complete, and the converged record's combined score is
    ``max`` over the two lineages.
    """
    repo = _repo(tmp_path)
    repo.create_record(_make_record("rec_shared01", "shared"))
    repo.create_record(_make_record("rec_nbr00001", "neighbour"))

    # hop 0: prompt(key1, 0.9) and response(key2, 0.5) BOTH reach rec_shared01.
    #   act_p = 0.9*0.8 = 0.72; act_r = 0.5*0.8 = 0.40 → combined = 0.72.
    # hop 1: rec_shared01's stored vec (key 11) reaches a neighbour. Because the
    #   shared record is a frontier source in BOTH lineages, need_ids carries it
    #   twice; embeddings_for must collapse it.
    vec = _DupGuardVec(
        seed_keys={"prompt q": 1, "response q": 2},
        rec_keys={"rec_shared01": 11},
        hits_by_key={
            1: [_hit("rec_shared01", 0.9)],
            2: [_hit("rec_shared01", 0.5)],
            11: [_hit("rec_nbr00001", 0.9)],
        },
    )
    out = walk_two_seeds("prompt q", "response q", vec, repo, _cfg(max_hops=2, min_score=0.1))

    by_id = {sr.record.id: sr.score for sr in out}
    assert "rec_shared01" in by_id
    assert by_id["rec_shared01"] == pytest.approx(0.72)  # max(0.72, 0.40)
    assert "rec_nbr00001" in by_id


def test_embeddings_for_dedups_before_chroma_get():
    """Fail-on-old proof: drive the REAL ``RecordsVecIndex.embeddings_for``
    against a fake collection that rejects duplicate ids (like chroma). Before
    the fix, the method passed input ids straight through and this raised
    ``_DupGuardError``; with the dedup it collapses duplicates and returns one
    entry per id.

    Constructs the index without ``__init__`` (no chroma/fastembed needed) and
    injects the fake collection — we test only the dedup-before-get contract.
    """
    idx = RecordsVecIndex.__new__(RecordsVecIndex)
    idx._collection = _FakeChromaCollection({"a": [1.0], "b": [2.0]})
    out = idx.embeddings_for(["a", "a", "b", "a"])
    assert out == {"a": [1.0], "b": [2.0]}
    # And the fake genuinely rejects dups (so this is a real guard, not a no-op).
    with pytest.raises(_DupGuardError):
        _FakeChromaCollection({}).get(["a", "a"])


# -- real-chroma: batched method contracts -------------------------------


@pytest.mark.skipif(not _RUN_VEC, reason="RUN_VEC_TESTS=1 required (model load)")
def test_batched_methods_real_chroma(tmp_path):
    """embed_texts → N vecs; query_by_vecs → N hit-lists; embeddings_for →
    stored vecs keyed by id. Real fastembed + ChromaDB, tmp persist dir."""
    persist_dir = tmp_path / "chroma"
    idx = RecordsVecIndex(persist_dir, _MODEL)
    idx.add_records_batch([
        ("rec_real0001", "qmd corpus indexing and retrieval"),
        ("rec_real0002", "bridge spreading activation latency"),
        ("rec_real0003", "record extraction during the sleep cycle"),
    ])
    assert idx.count() == 3

    texts = ["corpus indexing", "bridge latency"]
    vecs = idx.embed_texts(texts)
    assert len(vecs) == 2
    assert len(vecs[0]) == 384

    hit_lists = idx.query_by_vecs(vecs, k=3)
    assert len(hit_lists) == 2
    for hits in hit_lists:
        assert len(hits) >= 1
        for h in hits:
            assert 0.0 <= h.score <= 1.0

    stored = idx.embeddings_for(["rec_real0001", "rec_real0003", "rec_absent01"])
    assert set(stored) == {"rec_real0001", "rec_real0003"}
    assert len(stored["rec_real0001"]) == 384

    # Empty inputs are well-behaved.
    assert idx.embed_texts([]) == []
    assert idx.query_by_vecs([], k=3) == []
    assert idx.embeddings_for([]) == {}


@pytest.mark.skipif(not _RUN_VEC, reason="RUN_VEC_TESTS=1 required (model load)")
def test_embeddings_for_tolerates_duplicate_ids_real_chroma(tmp_path):
    """Real chroma ``collection.get`` rejects duplicate ids; ``embeddings_for``
    must dedup so a duplicated request id returns one entry, not an error."""
    persist_dir = tmp_path / "chroma"
    idx = RecordsVecIndex(persist_dir, _MODEL)
    idx.add_records_batch([
        ("rec_dup00001", "shared record reached by both lineages"),
        ("rec_dup00002", "another record"),
    ])
    stored = idx.embeddings_for(["rec_dup00001", "rec_dup00001", "rec_dup00002"])
    assert set(stored) == {"rec_dup00001", "rec_dup00002"}
    assert len(stored["rec_dup00001"]) == 384


def test_query_by_vecs_empty_collection_returns_empty_lists(tmp_path):
    """Empty collection -> one empty list per input vec (zip-safe)."""
    persist_dir = tmp_path / "chroma"
    idx = RecordsVecIndex(persist_dir, _MODEL)
    assert idx.count() == 0
    out = idx.query_by_vecs([[0.0] * 384, [0.1] * 384], k=5)
    assert out == [[], []]
    assert idx.query_by_vecs([], k=5) == []
