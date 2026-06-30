"""v0.7-x-vec-index W-A: fastembed + ChromaDB transport unit tests.

Two layers:

- fast unit tests that stub the embedder (deterministic vectors via SHA),
  so they run without the 28s model download and exercise the contract.
- ``@pytest.mark.slow`` real round-trip tests that load
  ``paraphrase-multilingual-MiniLM-L12-v2`` once; skipped unless
  ``RUN_VEC_TESTS=1`` (matches the spec convention §5 row 14).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from priming_stream.integrations.vec_index import RecordsVecIndex, VecHit


_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_RUN_VEC = os.environ.get("RUN_VEC_TESTS") == "1"


class _FakeEmbedder:
    """Deterministic 32-dim hash embedder; no model load."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def embed(self, texts):
        import numpy as np
        for t in texts:
            digest = hashlib.sha256(t.encode("utf-8")).digest()
            arr = np.frombuffer(digest, dtype=np.uint8).astype("float32")
            # Normalize to unit length so cosine distances are well-behaved.
            norm = np.linalg.norm(arr) or 1.0
            yield arr / norm


def _patch_embedder(idx: RecordsVecIndex) -> None:
    idx._model = _FakeEmbedder(idx._model_name)


# -- contract checks ------------------------------------------------------


def test_import_surface():
    # A2: import surface matches the contract.
    from priming_stream.integrations.vec_index import RecordsVecIndex, VecHit  # noqa: F401


def test_init_does_not_load_model(tmp_path: Path):
    # A3: __init__ does NOT load fastembed.
    idx = RecordsVecIndex(tmp_path / "chroma", _MODEL)
    assert idx._model is None


def test_collection_uses_cosine_space(tmp_path: Path):
    idx = RecordsVecIndex(tmp_path / "chroma", _MODEL)
    meta = idx._collection.metadata or {}
    assert meta.get("hnsw:space") == "cosine"


def test_count_empty(tmp_path: Path):
    idx = RecordsVecIndex(tmp_path / "chroma", _MODEL)
    assert idx.count() == 0


def test_search_empty_collection_returns_empty(tmp_path: Path):
    # A1: empty collection returns [].
    idx = RecordsVecIndex(tmp_path / "chroma", _MODEL)
    _patch_embedder(idx)
    assert idx.search("anything", k=5) == []


def test_has_record_missing(tmp_path: Path):
    idx = RecordsVecIndex(tmp_path / "chroma", _MODEL)
    assert idx.has_record("rec_missing") is False


def test_delete_missing_is_noop(tmp_path: Path):
    idx = RecordsVecIndex(tmp_path / "chroma", _MODEL)
    # Should not raise even if id is unknown.
    idx.delete_record("rec_unknown")
    assert idx.count() == 0


# -- stub-embedder round trips -------------------------------------------


def test_add_and_search_round_trip(tmp_path: Path):
    # A1: round-trip add+search returns the added record.
    idx = RecordsVecIndex(tmp_path / "chroma", _MODEL)
    _patch_embedder(idx)
    idx.add_record("rec_a", "Priming Stream architecture")
    assert idx.count() == 1
    assert idx.has_record("rec_a") is True

    hits = idx.search("Priming Stream architecture", k=3)
    assert len(hits) == 1
    assert isinstance(hits[0], VecHit)
    assert hits[0].record_id == "rec_a"
    assert hits[0].summary == "Priming Stream architecture"


def test_score_in_unit_interval(tmp_path: Path):
    # A1: cosine -> score in [0, 1] (clamped).
    idx = RecordsVecIndex(tmp_path / "chroma", _MODEL)
    _patch_embedder(idx)
    idx.add_records_batch([
        ("r1", "alpha beta gamma"),
        ("r2", "delta epsilon zeta"),
        ("r3", "eta theta iota"),
    ])
    hits = idx.search("alpha beta gamma", k=3)
    assert hits, "expected hits on populated collection"
    for h in hits:
        assert 0.0 <= h.score <= 1.0


def test_idempotent_add_upserts(tmp_path: Path):
    # A1: re-adding same id updates, doesn't duplicate.
    idx = RecordsVecIndex(tmp_path / "chroma", _MODEL)
    _patch_embedder(idx)
    idx.add_record("rec_a", "first version")
    idx.add_record("rec_a", "second version")
    assert idx.count() == 1
    hits = idx.search("second version", k=2)
    assert len(hits) == 1
    assert hits[0].summary == "second version"


# -- search_by_record (vector reuse for the spreading hot-path) -----------


def test_search_by_record_equivalent_to_search(tmp_path: Path):
    # Vector reuse: search_by_record(id) reuses the STORED embedding and must
    # return the SAME hits as search(that record's summary). The spreading walk
    # relies on this equivalence — identical results, no per-hop re-embedding.
    idx = RecordsVecIndex(tmp_path / "chroma", _MODEL)
    _patch_embedder(idx)
    idx.add_records_batch([
        ("r1", "alpha beta gamma"),
        ("r2", "delta epsilon zeta"),
        ("r3", "alpha beta something"),
    ])
    by_text = idx.search("alpha beta gamma", k=3)
    by_rec = idx.search_by_record("r1", k=3)
    assert [(h.record_id, round(h.score, 5)) for h in by_text] == \
           [(h.record_id, round(h.score, 5)) for h in by_rec]
    assert by_rec[0].record_id == "r1"  # the record is its own top hit


def test_search_by_record_missing_id_returns_empty(tmp_path: Path):
    idx = RecordsVecIndex(tmp_path / "chroma", _MODEL)
    _patch_embedder(idx)
    idx.add_record("r1", "alpha")
    assert idx.search_by_record("nope", k=3) == []


def test_search_by_record_empty_collection(tmp_path: Path):
    idx = RecordsVecIndex(tmp_path / "chroma", _MODEL)
    _patch_embedder(idx)
    assert idx.search_by_record("r1", k=3) == []


def test_delete_record(tmp_path: Path):
    # A1: deleted record gone.
    idx = RecordsVecIndex(tmp_path / "chroma", _MODEL)
    _patch_embedder(idx)
    idx.add_records_batch([("r1", "first"), ("r2", "second")])
    assert idx.count() == 2
    idx.delete_record("r1")
    assert idx.count() == 1
    assert idx.has_record("r1") is False
    assert idx.has_record("r2") is True


def test_persistence_across_instances(tmp_path: Path):
    # A1: writing with one client, reading with a fresh one at the same
    # persist_dir, recovers the data.
    persist = tmp_path / "chroma"
    idx1 = RecordsVecIndex(persist, _MODEL)
    _patch_embedder(idx1)
    idx1.add_records_batch([("r1", "alpha"), ("r2", "beta"), ("r3", "gamma")])
    assert idx1.count() == 3

    idx2 = RecordsVecIndex(persist, _MODEL)
    _patch_embedder(idx2)
    assert idx2.count() == 3
    assert idx2.has_record("r1")
    assert idx2.has_record("r2")
    assert idx2.has_record("r3")
    hits = idx2.search("alpha", k=3)
    ids = {h.record_id for h in hits}
    assert ids == {"r1", "r2", "r3"}


def test_add_records_batch_empty_is_noop(tmp_path: Path):
    idx = RecordsVecIndex(tmp_path / "chroma", _MODEL)
    _patch_embedder(idx)
    idx.add_records_batch([])
    assert idx.count() == 0


# -- real-fastembed round trip (slow) ------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not _RUN_VEC, reason="RUN_VEC_TESTS=1 required (model load)")
def test_real_fastembed_round_trip(tmp_path: Path):
    idx = RecordsVecIndex(tmp_path / "chroma", _MODEL)
    idx.add_records_batch([
        ("r_arch", "Priming Stream architecture System 1 substrate"),
        ("r_fert", "sex differences fertility civilizational decline"),
        ("r_smm", "SemNet positioning Software 3.0 Karpathy semantic mix"),
    ])
    assert idx.count() == 3

    hits = idx.search("substrate System 1 architecture", k=3)
    assert len(hits) == 3
    # Top hit should be the architecture record.
    assert hits[0].record_id == "r_arch"
    # All scores in [0, 1].
    for h in hits:
        assert 0.0 <= h.score <= 1.0
