"""Recency weighting (A.5b) + filter seam (A.5c) at bridge selection.

Pure, deterministic, I/O-free. Applies at SELECTION over bucket A only —
NOT in the per-hop spreading multiply (that would compound over hops and
distort the walk). Bucket B (lexical) is recency-EXEMPT (a named-old paper
must surface, not be demoted).

A.5b multiplier (locked formula):

    age_days  = max(0, now - source_date in days)   # future clamps to 0
    age_norm  = min(age_days / age_span_days, 1.0)
    f_recency = 1 - p_max * strength * age_norm      # bounds (1-p_max*strength, 1]

Undated records (index_cards / owner) and unparseable dates → f = 1.0,
neutral, never penalized. ``strength == 0`` → exact no-op.

A.5c is a hard cutoff seam: dated records older than a configured date are
dropped from bucket A; undated records always pass; an empty/unparseable
cutoff is off. Per-work-session granularity lives in the daemon config, not
here — this module just reads ``cfg.recency_filter_cutoff``.
"""
from __future__ import annotations

from datetime import datetime, timezone

from priming_stream.bridge.types import ScoredRecord


def _parse_dt(value: str | None) -> datetime | None:
    """Parse an ISO date/datetime to a tz-aware UTC datetime, or None.

    Accepts ``YYYY-MM-DDTHH:MM:SSZ`` and date-only ``YYYY-MM-DD`` (and the
    fractional-second variants ``source_date`` carries). Any parse failure —
    or a None input — returns None (treated as absent: f=1 / passes filter).
    Naive results are assumed UTC.
    """
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def f_recency(
    source_date: str | None,
    now: datetime,
    *,
    strength: float,
    age_span_days: int,
    p_max: float,
) -> float:
    """Recency multiplier in (1 - p_max*strength, 1]. See module docstring.

    ``now`` must be tz-aware UTC. Undated/unparseable ``source_date`` → 1.0.
    ``strength == 0`` → exactly 1.0 for any age. Monotonic: older → smaller.
    Future dates clamp to age 0 (f == 1.0); age beyond ``age_span_days``
    clamps to ``1 - p_max*strength``.
    """
    if strength == 0:
        return 1.0
    if age_span_days <= 0:
        return 1.0
    dt = _parse_dt(source_date)
    if dt is None:
        return 1.0
    age_days = max(0.0, (now - dt).total_seconds() / 86400.0)
    age_norm = min(age_days / age_span_days, 1.0)
    return 1.0 - p_max * strength * age_norm


def select_semantic(
    activated: list[ScoredRecord],
    cfg,
    *,
    now: datetime,
    exclude_recent_ids: frozenset[str] | set[str] = frozenset(),
) -> list[ScoredRecord]:
    """Recency-weight (A.5b) + cutoff-filter (A.5c) bucket A, then rank.

    For each record: ``rank_score = score * f_recency(source_date, ...)``.
    If ``cfg.recency_filter_cutoff`` is a parseable date, drop DATED records
    whose date is strictly before the cutoff (undated always pass). Sort by
    ``rank_score`` descending and truncate to ``bucket_total - bucket_lexical``.
    Returns new ScoredRecords carrying ``rank_score`` in ``.score``.

    ``exclude_recent_ids`` (item 3.3 cross-turn dedup): record ids primed in
    the last N turns of the same session. Dropped BEFORE truncation so the
    freed budget backfills from the recency-ranked tail — the queue advances
    and previously-crowded-out distal records surface. Empty default → no-op
    (all existing callers / MCP pull-bridges unchanged).
    """
    cutoff = _parse_dt(cfg.recency_filter_cutoff)

    scored: list[ScoredRecord] = []
    for sr in activated:
        if sr.record.id in exclude_recent_ids:
            continue  # 3.3 cross-turn dedup — filter before truncation
        if cutoff is not None:
            rec_dt = _parse_dt(sr.record.source_date)
            if rec_dt is not None and rec_dt < cutoff:
                continue
        f = f_recency(
            sr.record.source_date,
            now,
            strength=cfg.recency_strength,
            age_span_days=cfg.recency_age_span_days,
            p_max=cfg.recency_p_max,
        )
        scored.append(ScoredRecord(record=sr.record, score=sr.score * f))

    scored.sort(key=lambda s: s.score, reverse=True)
    budget = cfg.bucket_total - cfg.bucket_lexical
    return scored[:budget]
