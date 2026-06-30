"""Spreading activation over the records vec_index (v0.7-x-vec-index).

Active surface: ``walk_two_seeds`` — two-seed batched walk used by the
live bridge (daemon + MCP). The legacy single-seed ``spread()`` function
has been removed (W-G deletion pass).

Score handling: ``VecHit.score`` is already in [0, 1] (cosine-similarity-
like; computed as ``1.0 - distance`` against ChromaDB's cosine space),
so the walker uses ``a_in * hit.score * decay`` directly — no distance
inversion in this module.

Errors at the boundary: ``RecordsVecIndex`` methods may raise (fastembed
model load, ChromaDB IO). Callers are responsible for catching at the
hook/daemon boundary.
"""
from __future__ import annotations

from priming_stream.bridge.types import ScoredRecord
from priming_stream.core.graph_repo import GraphRepo
from priming_stream.core.models import Record
from priming_stream.integrations.vec_index import RecordsVecIndex, VecHit


def walk_two_seeds(
    prompt_text: str,
    prev_assistant_text: str,
    vec_index,
    repo,
    cfg,
) -> list[ScoredRecord]:
    """Two-seed batched spreading walk (A.1 — read-time bridge restructuring).

    Splits the seed into two independent activation lineages — the user
    ``prompt`` and the immediately-prior assistant turn (``response``) — so a
    short user pivot is not drowned by a long prior answer. Both lineages share
    identical machinery from ``hop_0 = 1.0``, so their raw accumulated
    activations are directly comparable; the combined per-record score is
    ``max(act_prompt, act_response)`` ("strong on either seed").

    The latency-critical invariant: **ONE batched chroma query per hop across
    BOTH lineages** (``vec_index.query_by_vecs``), not one per frontier entry.
    Two sequential single-seed walks (~2x) is the failure mode this avoids. The
    two seeds are also embedded in a single ``embed_texts`` call; hop>0 entries
    reuse each source record's STORED vector, fetched in one batched
    ``embeddings_for`` per hop.

    Returns ALL activated records (untruncated — selection truncates
    downstream) as :class:`ScoredRecord`, score = the RAW combined activation
    (no recency, no normalization), sorted by score descending.

    Empty/whitespace BOTH seeds → ``[]``. One empty seed → walk the other
    lineage only.
    """
    seeds: list[tuple[str, str]] = [
        (lineage, text)
        for lineage, text in (("prompt", prompt_text), ("response", prev_assistant_text))
        if text.strip()
    ]
    if not seeds:
        return []

    # Per-lineage accumulated activation: rec_id -> activation.
    act: dict[str, dict[str, float]] = {"prompt": {}, "response": {}}
    # Cache repo.get_record once per record (latency — avoid re-querying SQLite
    # on every hop and again at output assembly).
    rec_cache: dict[str, Record] = {}

    seed_vecs = vec_index.embed_texts([text for _, text in seeds])
    # Frontier entry: (lineage, query_vec_or_None, activation_in, source_id_or_None).
    # Seed entries carry their embedded query vector directly; hop>0 entries
    # carry None and the source record id, resolved via embeddings_for.
    frontier: list[tuple[str, list[float] | None, float, str | None]] = [
        (lineage, seed_vecs[i], 1.0, None) for i, (lineage, _) in enumerate(seeds)
    ]
    hop = 0

    while hop < cfg.max_hops and frontier:
        need_ids = [src for (_, qv, _, src) in frontier if qv is None and src]
        stored = vec_index.embeddings_for(need_ids) if need_ids else {}

        batch_vecs: list[list[float]] = []
        meta: list[tuple[str, float, str | None]] = []
        for lineage, qv, a_in, src in frontier:
            v = qv if qv is not None else stored.get(src)
            if v is None:
                continue  # missing stored vec -> skip this entry
            batch_vecs.append(v)
            meta.append((lineage, a_in, src))
        if not batch_vecs:
            break

        hit_lists = vec_index.query_by_vecs(batch_vecs, k=cfg.k_per_query)

        nxt: dict[str, list[tuple[str, None, float, str]]] = {"prompt": [], "response": []}
        for (lineage, a_in, src), hits in zip(meta, hit_lists):
            for hit in hits:
                rid = hit.record_id
                if not rid or rid == src:
                    continue
                a_new = a_in * hit.score * cfg.decay
                if a_new < cfg.min_score:
                    continue
                rec = rec_cache.get(rid)
                if rec is None:
                    rec = repo.get_record(rid)
                    if rec is None:
                        continue
                    rec_cache[rid] = rec
                act[lineage][rid] = act[lineage].get(rid, 0.0) + a_new
                nxt[lineage].append((lineage, None, a_new, rid))

        # Cap EACH lineage's frontier independently by activation_in, preserving
        # per-seed breadth (a single shared cap would let a strong lineage
        # starve the other).
        frontier = []
        for ln in ("prompt", "response"):
            frontier.extend(
                sorted(nxt[ln], key=lambda x: x[2], reverse=True)[:cfg.frontier_cap]
            )
        hop += 1

    out: list[ScoredRecord] = []
    for rid in set(act["prompt"]) | set(act["response"]):
        combined = max(act["prompt"].get(rid, 0.0), act["response"].get(rid, 0.0))
        rec = rec_cache.get(rid) or repo.get_record(rid)
        if rec is not None:
            out.append(ScoredRecord(record=rec, score=combined))
    out.sort(key=lambda sr: sr.score, reverse=True)
    return out
