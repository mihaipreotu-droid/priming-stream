"""Spread op for CLI/graph_ops surface (v0.7-x-vec-index Component A).

Uses ``walk_two_seeds(text, "")`` (single-seed: prompt only) — the same
A-pipeline as the hook and MCP handlers. Output carries ``source_date``
and ``kind`` for parity with search tools; capped at
``bridge_cfg.max_records`` (the raw-spread output cap documented in
config.py for deliberate MCP/CLI surfaces).

``graph_spread_op`` is an internal op (used by diagnostics and tests; no
``prime graph spread`` CLI command is registered). The deliberate
user-facing surfaces are the MCP ``graph_spread`` tool and ``prime search``.
"""
from __future__ import annotations

from priming_stream.bridge.spreading import walk_two_seeds
from priming_stream.core.graph_repo import GraphRepo
from priming_stream.integrations.vec_index import RecordsVecIndex


def graph_spread_op(
    text: str,
    vec_index: RecordsVecIndex,
    repo: GraphRepo,
    bridge_cfg,
) -> list[dict]:
    """Run the A-pipeline walk on ``text`` and return ranked record dicts.

    Single-seed walk (prompt=text, prev=""). Capped at
    ``bridge_cfg.max_records``. Each item: ``{record_id, summary, rank,
    source_date, kind}``.
    """
    if not text.strip():
        return []
    activated = walk_two_seeds(
        text, "", vec_index=vec_index, repo=repo, cfg=bridge_cfg,
    )
    top = activated[: bridge_cfg.max_records]
    return [
        {
            "record_id": sr.record.id,
            "summary": sr.record.summary,
            "rank": i + 1,
            "source_date": sr.record.source_date,
            "kind": sr.record.kind,
        }
        for i, sr in enumerate(top)
    ]
