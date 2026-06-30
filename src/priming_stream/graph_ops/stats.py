"""``graph_stats`` op for the MCP tool surface (v0.7-x-vec-index).

Returns the new v0.7-x-vec-index shape::

    {
      "records_count": int,
      "last_sleep_cycle": {...} | None,
      "vec_index_size": int,
    }

``vec_index_size`` replaces the old ``qmd_collections`` list. The vector
index is now in-process (fastembed + ChromaDB), so ``count()`` is a
cheap call against the persisted collection.

If ``vec_index.count`` raises (corrupt persist dir, missing collection),
we record ``vec_index_size = None`` rather than crashing the tool — the
SQLite mirror is the source of truth and the dashboard surfaces the
degraded state.
"""
from __future__ import annotations

from priming_stream.core.graph_repo import GraphRepo
from priming_stream.integrations.vec_index import RecordsVecIndex


def graph_stats(
    repo: GraphRepo,
    vec_index: RecordsVecIndex,
) -> dict:
    records_count = len(repo.list_records())
    cycles = repo.list_sleep_cycles(limit=1)
    last_cycle = cycles[0] if cycles else None

    try:
        vec_size: int | None = vec_index.count()
    except Exception:  # noqa: BLE001 - degrade gracefully
        vec_size = None

    return {
        "records_count": records_count,
        "last_sleep_cycle": last_cycle,
        "vec_index_size": vec_size,
    }
