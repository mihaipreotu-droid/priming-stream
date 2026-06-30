"""v0.7-x W-C: heuristic record-extraction filter."""
from __future__ import annotations

from pathlib import Path

import pytest

from priming_stream.ingest.record_filter import THRESHOLD, load_markers, score


REPO_ROOT = Path(__file__).resolve().parents[2]
MARKERS_PATH = REPO_ROOT / "config" / "record_markers.toml"


@pytest.fixture(scope="module")
def markers() -> dict:
    return load_markers(MARKERS_PATH)


# -- load_markers --------------------------------------------------------


def test_load_markers_shape(markers):
    assert "ro" in markers
    assert "en" in markers
    assert "decision" in markers["ro"]
    assert "decision" in markers["en"]
    assert isinstance(markers["ro"]["decision"], list)
    assert all(isinstance(m, str) for m in markers["ro"]["decision"])


def test_threshold_constant():
    assert THRESHOLD == 0.3


# -- score ---------------------------------------------------------------


def test_score_neutral_text(markers):
    s, hits = score("hello world", markers)
    assert s == 0.0
    assert hits == {}


def test_score_empty_text(markers):
    s, hits = score("", markers)
    assert s == 0.0
    assert hits == {}


def test_score_ro_decision_passes_threshold(markers):
    s, hits = score("decid să folosesc qmd peste graf-ul vechi", markers)
    assert s >= THRESHOLD
    assert "decision" in hits


def test_score_en_decision_passes_threshold(markers):
    s, hits = score("Let's decide on the new approach", markers)
    assert s >= THRESHOLD
    assert "decision" in hits


def test_score_outcome_marker(markers):
    s, hits = score("the migration broke, rolled back", markers)
    assert s >= THRESHOLD
    assert "outcome" in hits


def test_score_mixed_intents_groups_hits(markers):
    text = (
        "Initial outcome: it worked. Then I realize the tradeoff between "
        "speed and clarity — let's decide tomorrow."
    )
    s, hits = score(text, markers)
    assert s >= THRESHOLD
    # decision (let's decide / decide / DECIS variants), outcome (worked),
    # insight (I realize), contrast (tradeoff) all present.
    assert "decision" in hits
    assert "outcome" in hits
    assert "insight" in hits
    assert "contrast" in hits


def test_score_case_insensitive(markers):
    s_lower, _ = score("decid imediat", markers)
    s_upper, _ = score("DECID IMEDIAT", markers)
    assert s_lower == s_upper
    assert s_lower >= THRESHOLD


def test_score_saturates_at_one(markers):
    # Many marker hits should still cap at 1.0.
    text = (
        "decid, alegem, going with, let's use, DECIS, decide, "
        "a mers, worked, broke, fixed, renunț, drop, switch, "
        "tradeoff, vs, diferența, equivalent, depinde de"
    )
    s, _hits = score(text, markers)
    assert s == 1.0


def test_score_single_hit_threshold_pass(markers):
    s, hits = score("merg cu varianta simplă", markers)
    # ``merg cu`` is a decision marker; one hit -> 0.3.
    assert s == pytest.approx(0.3)
    assert "decision" in hits
