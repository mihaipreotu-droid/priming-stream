"""fastembed + ChromaDB transport for the records substrate (v0.7-x-vec-index).

Replaces the qmd CLI transport: bridge spreading walk and sleep-cycle
writes both go through this module. Collection ``records`` is opened
(or created) on a single ChromaDB :class:`PersistentClient`; embeddings
are computed locally via :class:`fastembed.TextEmbedding` (model from
``config.vec_index.model_name``; bge-m3 int8, 1024-dim, since 2026-07-21)
so we never call out to a remote API.

Two conventions are load-bearing:

1. **Lazy model init.** ``__init__`` opens ChromaDB but does NOT load
   fastembed. ``_get_model()`` populates from a process-level cache on
   first call. Tests assert this contract; consumers that only need
   ``count()`` / ``has_record()`` / ``delete_record()`` pay zero
   embedder cost.
2. **Cosine space.** The collection is created with
   ``metadata={"hnsw:space": "cosine"}`` so the distance returned by
   ``collection.query`` lives in [0, 2] for unit-norm vectors and
   ``score = 1.0 - distance`` lands in [0, 1] (clamped for FP slack).

**Process-level model singleton.** A module-level cache keyed by
``model_name`` holds at most one :class:`fastembed.TextEmbedding` per
model across the lifetime of the process. Second-and-later
:class:`RecordsVecIndex` instances (e.g. the one constructed by
``/v1/reload`` to atomic-swap the daemon's index) reuse the same
underlying ONNX session — reload drops from ~1.5-2.5s to ~25ms. The
cache populate path is guarded by a ``threading.Lock`` so concurrent
first-touches from a ``ThreadingHTTPServer`` worker pool don't double-
init. Existing test patches that monkey-patch ``idx._model`` directly
are unaffected (instance attribute shadows the cache).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ``chromadb`` is imported lazily inside ``RecordsVecIndex.__init__`` (item 3.4):
# it costs ~1.5s to import and pulls a large dependency tree. Deferring it to
# construction keeps merely IMPORTING ``RecordsVecIndex`` (the class) cheap, so
# the read-only MCP server — whose verify tool (graph_chunk_around_anchor) never
# builds an index — starts its stdio handshake fast instead of paying chromadb
# up front. The daemon still constructs an index at startup, so its warm-model
# behaviour is unchanged.


_COLLECTION_NAME = "records"

# Process-level cache: at most one TextEmbedding per model_name across the
# whole process. Populated lazily by _get_model() under _MODEL_CACHE_LOCK.
# See module docstring for rationale.
_MODEL_CACHE: dict[str, Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()

# Custom fastembed models not in the stock list. Registered lazily (right
# before a TextEmbedding is built, under _MODEL_CACHE_LOCK) so merely
# importing this module stays cheap — the read-only MCP server never reaches
# _get_model().
#
# BGE-M3 is stored in ONNX external-data format: model.onnx (~0.7MB graph
# skeleton) + model.onnx_data (~2.2GB weights). ``additional_files`` is
# LOAD-BEARING: omit it and fastembed fetches only the skeleton, yielding a
# weightless model that loads WITHOUT error and emits garbage embeddings
# (silent corruption). See docs/embedding-options.md.
_REGISTERED_MODELS: set[str] = set()

# Custom model specs, registered ON DEMAND for the model actually configured
# (2026-07-21 review; — both used to register unconditionally; the
# unused one is now registered only if a rollback instance asks for it).
#
# fp32 note: BGE-M3 is stored in ONNX external-data format; ``additional_files``
# is LOAD-BEARING there (omit it → weightless skeleton → garbage embeddings,
# see docs/embedding-options.md). The int8 file is
# self-contained (~568MB) — the weightless trap applies only to fp32.
#
# int8 note (adopted 2026-07-21 after a quality + latency probe): HF
# revision 25b9af8e sits pinned in the local hub cache; fastembed cannot pin
# revisions, so a cache wipe re-fetches main. The daemon's warmup canary
# (daemon/server.py:_run_canary) gates every load against a wrong artifact.
_CUSTOM_MODEL_SPECS: dict[str, dict] = {
    "BAAI/bge-m3": dict(
        source="BAAI/bge-m3",
        model_file="onnx/model.onnx",
        additional_files=["onnx/model.onnx_data"],
    ),
    "onnx-community/bge-m3-ONNX-int8": dict(
        source="onnx-community/bge-m3-ONNX",
        model_file="onnx/model_int8.onnx",
        additional_files=None,
    ),
}


def _ensure_custom_models_registered(model_name: str | None = None) -> None:
    """Register the custom fastembed model(s) needed (idempotent).

    ``model_name`` given → register only that one (if it is a custom spec);
    ``None`` → register all specs (back-compat for callers that don't know
    the target yet). Must run before ``TextEmbedding(model_name=...)`` for a
    custom model — they are absent from fastembed's stock list. Callers hold
    ``_MODEL_CACHE_LOCK`` so this need not lock itself.
    """
    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource, PoolingType

    wanted = [model_name] if model_name else list(_CUSTOM_MODEL_SPECS)
    todo = [m for m in wanted
            if m in _CUSTOM_MODEL_SPECS and m not in _REGISTERED_MODELS]
    if not todo:
        return
    stock = {m["model"] for m in TextEmbedding.list_supported_models()}
    for name in todo:
        spec = _CUSTOM_MODEL_SPECS[name]
        if name not in stock:
            TextEmbedding.add_custom_model(
                model=name,
                pooling=PoolingType.CLS,
                normalization=True,
                sources=ModelSource(hf=spec["source"]),
                dim=1024,
                model_file=spec["model_file"],
                additional_files=spec["additional_files"],
            )
        _REGISTERED_MODELS.add(name)


@dataclass
class VecHit:
    record_id: str
    score: float
    summary: str


class RecordsVecIndex:
    def __init__(self, persist_dir: Path, model_name: str) -> None:
        import chromadb  # lazy: see module-level note (item 3.4 MCP startup)

        persist_dir.mkdir(parents=True, exist_ok=True)
        self._persist_dir = persist_dir
        self._model_name = model_name
        self._model: Any | None = None
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            _COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # -- write path -------------------------------------------------------

    def add_record(self, record_id: str, summary: str) -> None:
        vec = self._embed_one(summary)
        self._collection.upsert(
            ids=[record_id],
            documents=[summary],
            embeddings=[vec],
        )

    def add_records_batch(self, items: list[tuple[str, str]]) -> None:
        if not items:
            return
        ids = [rid for rid, _ in items]
        docs = [summary for _, summary in items]
        vecs = self._embed_many(docs)
        self._collection.upsert(ids=ids, documents=docs, embeddings=vecs)

    def delete_record(self, record_id: str) -> None:
        self._collection.delete(ids=[record_id])

    # -- read path --------------------------------------------------------

    def search(self, query_text: str, k: int) -> list[VecHit]:
        if self._collection.count() == 0:
            return []
        qvec = self._embed_one(query_text)
        return self._query_by_vec(qvec, k)

    def search_by_record(self, record_id: str, k: int) -> list[VecHit]:
        """Like :meth:`search` but reuse the record's STORED embedding instead
        of re-embedding text — the spreading walk's multi-hop frontier queries
        with record *summaries* that are already embedded in this collection,
        so this skips the per-hop fastembed call (the bridge hot-path cost).

        Returns ``[]`` if ``record_id`` is absent (or has no embedding) — the
        caller falls back to text search. The record itself comes back as the
        top hit (its own nearest neighbour); the spread filters it out.
        """
        if self._collection.count() == 0:
            return []
        got = self._collection.get(ids=[record_id], include=["embeddings"])
        embs = got.get("embeddings")
        if embs is None or len(embs) == 0 or embs[0] is None:
            return []
        return self._query_by_vec(embs[0], k)

    def _query_by_vec(self, qvec, k: int) -> list[VecHit]:
        res = self._collection.query(query_embeddings=[qvec], n_results=k)
        ids = (res.get("ids") or [[]])[0]
        documents = (res.get("documents") or [[]])[0]
        distances = (res.get("distances") or [[]])[0]
        hits: list[VecHit] = []
        for rid, doc, dist in zip(ids, documents, distances):
            score = max(0.0, min(1.0, 1.0 - float(dist)))
            hits.append(VecHit(record_id=rid, score=score, summary=doc or ""))
        return hits

    # -- read path (batched, A.1 two-seed walk) ---------------------------

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Public batched embed (one fastembed pass for all texts).

        Wraps the existing :meth:`_embed_many` so the two-seed walk can embed
        both seeds in a single model call. Empty input → empty list.
        """
        if not texts:
            return []
        return self._embed_many(texts)

    def embeddings_for(self, record_ids: list[str]) -> dict[str, list[float]]:
        """Batched stored-vector fetch: ``id -> embedding`` for the given ids.

        One ``collection.get`` for the whole batch (the per-hop frontier
        re-queries by each source record's STORED vector — this avoids N
        single-id gets). Missing ids are simply absent from the dict. Chroma
        may reorder the returned rows, so the result is mapped by the returned
        id, not by request position.

        Tolerates duplicate input ids: the two-seed walk legitimately produces
        the same source record in both lineages' frontiers, but chroma's
        ``get`` raises on duplicate ids, so they are collapsed (order-preserving)
        before the get. The result keys by id, so duplicates are harmless.
        """
        if not record_ids:
            return {}
        record_ids = list(dict.fromkeys(record_ids))
        got = self._collection.get(ids=record_ids, include=["embeddings"])
        ids = got.get("ids") or []
        embs = got.get("embeddings")
        if embs is None:
            return {}
        out: dict[str, list[float]] = {}
        for rid, emb in zip(ids, embs):
            if emb is None:
                continue
            out[rid] = emb.tolist() if hasattr(emb, "tolist") else list(emb)
        return out

    def query_by_vecs(self, vecs: list[list[float]], k: int) -> list[list[VecHit]]:
        """ONE batched chroma query for many vectors → one VecHit list each.

        ``collection.query(query_embeddings=vecs, n_results=k)`` returns
        ids/documents/distances as lists-of-lists (one inner list per input
        vec); the returned lists are in input order. This is the single chroma
        round-trip per hop that keeps the two-seed walk ≈ one walk.

        Empty collection or empty ``vecs`` → a flat list of empty lists (one
        per input vec, so callers can ``zip`` against their metadata).
        """
        if not vecs:
            return []
        if self._collection.count() == 0:
            return [[] for _ in vecs]
        res = self._collection.query(query_embeddings=vecs, n_results=k)
        ids_lists = res.get("ids") or []
        doc_lists = res.get("documents") or []
        dist_lists = res.get("distances") or []
        out: list[list[VecHit]] = []
        for i in range(len(vecs)):
            ids = ids_lists[i] if i < len(ids_lists) else []
            docs = doc_lists[i] if i < len(doc_lists) else []
            dists = dist_lists[i] if i < len(dist_lists) else []
            hits: list[VecHit] = []
            for rid, doc, dist in zip(ids, docs, dists):
                score = max(0.0, min(1.0, 1.0 - float(dist)))
                hits.append(VecHit(record_id=rid, score=score, summary=doc or ""))
            out.append(hits)
        return out

    # -- maintenance ------------------------------------------------------

    def count(self) -> int:
        return self._collection.count()

    def has_record(self, record_id: str) -> bool:
        got = self._collection.get(ids=[record_id])
        return len(got.get("ids", [])) > 0

    # -- internals --------------------------------------------------------

    def _get_model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            with _MODEL_CACHE_LOCK:
                _ensure_custom_models_registered(self._model_name)
                cached = _MODEL_CACHE.get(self._model_name)
                if cached is None:
                    cached = TextEmbedding(model_name=self._model_name)
                    _MODEL_CACHE[self._model_name] = cached
                self._model = cached
        return self._model

    def _embed_one(self, text: str) -> list[float]:
        vecs = list(self._get_model().embed([text]))
        return vecs[0].tolist()

    def _embed_many(self, texts: list[str]) -> list[list[float]]:
        vecs = list(self._get_model().embed(texts))
        return [v.tolist() for v in vecs]
