"""graph_ops — read-only operation layer over the v0.7-x-vec-index substrate.

v0.7-x: nodes/edges/path/spread are gone. v0.7-x-vec-index: the qmd
chunks-search escape hatch (``graph_search_chunks``) is also dropped —
chunks are reachable from records via :func:`graph_chunk_around_anchor`
(tier-2 verification path). Mass chunk search wasn't a stated requirement.

The active surface re-exports the v0.7-x-vec-index functions used by the
MCP tools.
"""
from __future__ import annotations

from priming_stream.graph_ops.records_search import (
    graph_chunk_around_anchor,
    graph_records,
    graph_search_lexical,
    graph_search_records,
)
from priming_stream.graph_ops.spread_op import graph_spread_op
from priming_stream.graph_ops.stats import graph_stats

__all__ = [
    "graph_search_records",
    "graph_search_lexical",
    "graph_records",
    "graph_chunk_around_anchor",
    "graph_spread_op",
    "graph_stats",
]
