"""Tests for v0.7-x Component A recency weighting (A.5b) + filter seam (A.5c).

Covers bridge/recency.py: f_recency (multiplier formula, bounds, no-op,
undated/unparseable neutrality, monotonicity, clamps) and select_semantic
(recency-weighted ranking, truncation, strength=0 no-op, A.5c cutoff).
"""
from __future__ import annotations

from datetime import datetime, timezone

from priming_stream.bridge.recency import f_recency, select_semantic
from priming_stream.bridge.types import ScoredRecord
from priming_stream.core.config import BridgeConfig
from priming_stream.core.models import Record

NOW = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _rec(rid: str, source_date: str | None) -> Record:
    return Record(
        id=rid,
        source_uri="qmd://x",
        anchor_offset_start=0,
        anchor_offset_end=1,
        summary=f"summary {rid}",
        created_at="2026-06-02T00:00:00Z",
        source_date=source_date,
    )


def _days_ago(n: int) -> str:
    """ISO source_date n days before NOW, with 'Z' suffix."""
    from datetime import timedelta

    return (NOW - timedelta(days=n)).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- f_recency -------------------------------------------------------------

def test_strength_zero_is_exact_noop():  # (a)
    for age in (0, 10, 100, 180, 365, 10_000):
        assert f_recency(_days_ago(age), NOW, strength=0.0,
                         age_span_days=180, p_max=0.5) == 1.0


def test_undated_is_neutral():  # (b)
    assert f_recency(None, NOW, strength=0.25, age_span_days=180, p_max=0.5) == 1.0


def test_unparseable_is_neutral():  # (c)
    assert f_recency("garbage", NOW, strength=0.25,
                     age_span_days=180, p_max=0.5) == 1.0
    assert f_recency("2026-13-99", NOW, strength=0.25,
                     age_span_days=180, p_max=0.5) == 1.0
    assert f_recency("", NOW, strength=0.25,
                     age_span_days=180, p_max=0.5) == 1.0


def test_bounds():  # (d)
    p_max, strength = 0.5, 0.25
    floor = 1.0 - p_max * strength  # 0.875
    # Strict lower bound holds for ages strictly below the span; the floor is
    # ATTAINED only at/beyond the span (AC g, age_norm clamps to 1.0). So the
    # full-range invariant is floor <= f <= 1.0, strict below the span.
    for age in (0, 1, 5, 30, 90, 179):
        f = f_recency(_days_ago(age), NOW, strength=strength,
                      age_span_days=180, p_max=p_max)
        assert floor < f <= 1.0
    for age in (180, 200, 500, 5000):
        f = f_recency(_days_ago(age), NOW, strength=strength,
                      age_span_days=180, p_max=p_max)
        assert floor <= f <= 1.0


def test_monotonic_in_age():  # (e)
    f10 = f_recency(_days_ago(10), NOW, strength=0.25, age_span_days=180, p_max=0.5)
    f100 = f_recency(_days_ago(100), NOW, strength=0.25, age_span_days=180, p_max=0.5)
    assert f10 >= f100
    # strictly older is strictly smaller below the span clamp
    assert f10 > f100


def test_future_clamps_to_one():  # (f)
    future = (NOW.replace(year=NOW.year + 1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert f_recency(future, NOW, strength=0.25,
                     age_span_days=180, p_max=0.5) == 1.0
    # exactly now -> age 0 -> 1.0
    assert f_recency(_days_ago(0), NOW, strength=0.25,
                     age_span_days=180, p_max=0.5) == 1.0


def test_beyond_span_clamps_to_floor():  # (g)
    p_max, strength = 0.5, 0.25
    floor = 1.0 - p_max * strength
    f365 = f_recency(_days_ago(365), NOW, strength=strength,
                     age_span_days=180, p_max=p_max)
    assert f365 == floor
    # exactly at the span boundary already hits the floor
    f180 = f_recency(_days_ago(180), NOW, strength=strength,
                     age_span_days=180, p_max=p_max)
    assert f180 == floor


def test_date_only_format_parses():
    # date-only YYYY-MM-DD must parse (not be treated as absent)
    f = f_recency("2026-01-01", NOW, strength=0.25, age_span_days=180, p_max=0.5)
    assert f < 1.0  # ~150 days old -> penalized, so it parsed


# --- select_semantic -------------------------------------------------------

def _sr(rid: str, source_date: str | None, score: float) -> ScoredRecord:
    return ScoredRecord(record=_rec(rid, source_date), score=score)


def test_truncates_to_semantic_budget_default():  # (h)
    cfg = BridgeConfig()  # bucket_total=25, bucket_lexical=5 -> budget 20
    items = [_sr(f"rec_{i}", _days_ago(i), 1.0 - i * 0.001) for i in range(30)]
    out = select_semantic(items, cfg, now=NOW)
    assert len(out) == 20


def test_truncates_to_small_budget():  # (h, small cfg)
    cfg = BridgeConfig(bucket_total=5, bucket_lexical=2)  # budget 3
    items = [_sr(f"rec_{i}", None, 1.0 - i * 0.01) for i in range(10)]
    out = select_semantic(items, cfg, now=NOW)
    assert len(out) == 3
    assert [s.record.id for s in out] == ["rec_0", "rec_1", "rec_2"]


def test_recency_multiply_changes_ranking():  # (i)
    # old record has slightly higher raw; recent record has slightly lower raw.
    # At gentle strength the recency multiply flips them when ages differ enough.
    cfg = BridgeConfig(recency_strength=0.25, recency_age_span_days=180,
                       recency_p_max=0.5, bucket_total=10, bucket_lexical=0)
    old_high = _sr("old", _days_ago(180), 0.90)    # f = 0.875 -> 0.7875
    new_low = _sr("new", _days_ago(0), 0.85)       # f = 1.0   -> 0.85
    out = select_semantic([old_high, new_low], cfg, now=NOW)
    assert [s.record.id for s in out] == ["new", "old"]  # ranking flipped
    # scores carry rank_score
    assert abs(out[0].score - 0.85) < 1e-9
    assert abs(out[1].score - 0.7875) < 1e-9


def test_recency_does_not_flip_when_gap_too_large():  # (i, control)
    # if the old record's raw lead exceeds the gentle recency penalty, it holds.
    cfg = BridgeConfig(recency_strength=0.25, recency_age_span_days=180,
                       recency_p_max=0.5, bucket_total=10, bucket_lexical=0)
    old_high = _sr("old", _days_ago(180), 0.90)    # -> 0.7875
    new_low = _sr("new", _days_ago(0), 0.80)       # -> 0.80
    out = select_semantic([old_high, new_low], cfg, now=NOW)
    assert [s.record.id for s in out] == ["new", "old"]  # 0.80 > 0.7875
    # widen the lead so old wins
    old_higher = _sr("old", _days_ago(180), 0.95)  # -> 0.83125
    out2 = select_semantic([old_higher, new_low], cfg, now=NOW)
    assert [s.record.id for s in out2] == ["old", "new"]  # 0.83125 > 0.80


def test_strength_zero_ranking_identical_to_raw():  # (j)
    cfg = BridgeConfig(recency_strength=0.0, bucket_total=10, bucket_lexical=0)
    # old records with higher raw must stay on top despite age (no-op)
    items = [
        _sr("a_old_high", _days_ago(300), 0.9),
        _sr("b_new_mid", _days_ago(1), 0.7),
        _sr("c_old_low", _days_ago(300), 0.5),
    ]
    out = select_semantic(items, cfg, now=NOW)
    assert [s.record.id for s in out] == ["a_old_high", "b_new_mid", "c_old_low"]
    # raw scores preserved exactly
    assert [s.score for s in out] == [0.9, 0.7, 0.5]


def test_cutoff_empty_no_filtering():  # (k)
    cfg = BridgeConfig(recency_filter_cutoff="", bucket_total=10, bucket_lexical=0)
    items = [_sr("old", "2026-01-01", 0.9), _sr("new", "2026-05-01", 0.8)]
    out = select_semantic(items, cfg, now=NOW)
    assert {s.record.id for s in out} == {"old", "new"}


def test_cutoff_unparseable_treated_as_off():
    cfg = BridgeConfig(recency_filter_cutoff="not-a-date",
                       bucket_total=10, bucket_lexical=0)
    items = [_sr("old", "2026-01-01", 0.9), _sr("new", "2026-05-01", 0.8)]
    out = select_semantic(items, cfg, now=NOW)
    assert {s.record.id for s in out} == {"old", "new"}


def test_cutoff_drops_dated_older_keeps_undated_and_newer():  # (l)
    cfg = BridgeConfig(recency_filter_cutoff="2026-03-01", recency_strength=0.0,
                       bucket_total=10, bucket_lexical=0)
    items = [
        _sr("dated_older", "2026-02-15T00:00:00Z", 0.9),  # dropped
        _sr("dated_newer", "2026-04-01T00:00:00Z", 0.8),  # kept
        _sr("undated", None, 0.7),                         # always kept
        _sr("dated_at_cutoff", "2026-03-01T09:00:00Z", 0.6),  # same day -> not < cutoff -> kept
    ]
    out = select_semantic(items, cfg, now=NOW)
    ids = {s.record.id for s in out}
    assert "dated_older" not in ids
    assert ids == {"dated_newer", "undated", "dated_at_cutoff"}


def test_empty_input_returns_empty():
    cfg = BridgeConfig()
    assert select_semantic([], cfg, now=NOW) == []
