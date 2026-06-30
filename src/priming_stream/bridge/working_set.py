"""Bridge orchestrator — the A-pipeline (v0.7-x Component A integration).

Composes the four frozen leaf modules into one read-time priming pass:

    walk_two_seeds          (A.1) -> raw combined activation, untruncated
    select_semantic         (A.5b/A.5c, bucket A) -> recency-weighted top-N
    lexical_bucket          (A.2, bucket B) -> FTS5 BM25, A-first deduped

Exactly ONE embedding walk per call (``walk_two_seeds`` issues the only
vec-index queries). Recency is O(activated) arithmetic; the lexical bucket is
one FTS5 query over the prompt. No second embed, no second walk — the latency
shape is one-walk + cheap-tail, matching the §5.1 gate.

Pure composition: no I/O of its own beyond what the leaves do, no LLM, no
mutation of the substrate (read-only over records). The daemon and the hook
call this; tests build it with a real tmp SQLite + a stub vec_index.

``priming_items`` converts a ``PrimingResult`` to the two plain-dict lists
expected by ``daemon.render.render_buckets`` (and any other caller that
needs the daemon's HTTP serialisation shape). This is the shared conversion
that daemon/server.py and mcp_server/tools.py both import — single source of
truth for the ``{record_id, summary, rank, source_date, kind, …}`` dict shape.
"""
from __future__ import annotations

from datetime import datetime, timezone

from priming_stream.bridge.lexical import lexical_bucket
from priming_stream.bridge.recency import select_semantic
from priming_stream.bridge.spreading import walk_two_seeds
from priming_stream.bridge.types import PrimingResult, ScoredRecord


def _scored_to_item(sr: ScoredRecord, rank: int) -> dict:
    """Serialize one :class:`ScoredRecord` to a per-record response dict.

    ``rank`` is 1-based within its own bucket. Fields match the daemon
    HTTP shape so callers (daemon, MCP) can share this converter.
    """
    r = sr.record
    return {
        "record_id": r.id,
        "summary": r.summary,
        "rank": rank,
        "source_uri": r.source_uri,
        "anchor_start": r.anchor_offset_start or 0,
        "anchor_end": r.anchor_offset_end or 0,
        "source_date": r.source_date,
        "kind": r.kind,
    }


def priming_items(
    result: PrimingResult,
) -> tuple[list[dict], list[dict]]:
    """Convert a :class:`PrimingResult` to two plain-dict lists.

    Returns ``(semantic_items, lexical_items)`` where each item has the
    shape ``{record_id, summary, rank, source_uri, anchor_start,
    anchor_end, source_date, kind}`` — the same shape the daemon HTTP
    endpoint emits and ``daemon.render.render_buckets`` expects.

    Used by daemon/server.py, mcp_server/tools.py, and tests. The daemon
    keeps its own ``_scored_to_item`` reference pointing here for
    backward-compatibility.
    """
    sem = [_scored_to_item(sr, i + 1) for i, sr in enumerate(result.semantic)]
    lex = [_scored_to_item(sr, i + 1) for i, sr in enumerate(result.lexical)]
    return sem, lex


def build_priming(
    prompt: str,
    prev: str,
    *,
    vec_index,
    repo,
    conn,
    cfg,
    now: datetime | None = None,
) -> PrimingResult:
    """Run the A-pipeline and return the two priming buckets.

    - ``semantic`` (bucket A): ``select_semantic`` over the raw two-seed walk —
      recency-weighted (A.5b), cutoff-filtered (A.5c), ranked, and truncated to
      ``cfg.bucket_total - cfg.bucket_lexical``.
    - ``lexical`` (bucket B): ``lexical_bucket`` over the USER PROMPT ONLY,
      BM25-ordered, A-first deduped against the semantic ids, capped at
      ``cfg.bucket_lexical``, ``index_card``-biased.

    ``now`` defaults to the current UTC instant (recency reads tz-aware UTC).
    """
    if now is None:
        now = datetime.now(timezone.utc)

    semantic = select_semantic(
        walk_two_seeds(prompt, prev, vec_index, repo, cfg),
        cfg,
        now=now,
    )
    exclude = {sr.record.id for sr in semantic}
    lexical = lexical_bucket(
        conn,
        prompt,
        limit=cfg.bucket_lexical,
        exclude_ids=exclude,
        kind_bias=True,
    )
    return PrimingResult(semantic=semantic, lexical=lexical)
