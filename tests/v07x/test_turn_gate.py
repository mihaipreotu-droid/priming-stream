"""P2/P3 turn-gate — unit tests (final amendment: FULL or WHISPER, no silence).

Gate logic in ``build_priming`` (whisper-floor / whisper-regime /
whisper-notification / kickoff / off / no features) with stubbed
walk+selection, plus the hook-side helpers. The default knob values were
calibrated empirically — these tests pin mechanics, not the constants.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import priming_stream.bridge.working_set as ws
from priming_stream.bridge.types import PrimingResult, ScoredRecord
from priming_stream.hooks.user_prompt_submit import (
    _WEAK_FIELD_MARKER,
    _is_notification_turn,
    _recent_tool_density,
)


def _rec(rid, score):
    r = SimpleNamespace(id=rid, summary=f"summary {rid}", source_uri="qmd://t",
                        anchor_offset_start=0, anchor_offset_end=1,
                        source_date=None, kind="claim")
    return ScoredRecord(record=r, score=score)


def _cfg(**over):
    base = dict(bucket_lexical=5, turn_floor=0.40, regime_density=0.6,
                whisper_k=5, whisper_lex_k=3, kickoff_turns=3)
    base.update(over)
    return SimpleNamespace(**base)


def _run(monkeypatch, cfg, scores, features, capture=None):
    sems = [_rec(f"rec_{i:08d}", s) for i, s in enumerate(scores)]
    monkeypatch.setattr(ws, "walk_two_seeds", lambda *a, **k: sems)
    monkeypatch.setattr(ws, "select_semantic", lambda walk, c, **k: walk)

    def _fake_lexical(conn, prompt, *, limit, **k):
        if capture is not None:
            capture["lex_limit"] = limit
        return [_rec(f"rec_lex{i:05d}", 1.0) for i in range(limit)]
    monkeypatch.setattr(ws, "lexical_bucket", _fake_lexical)
    return ws.build_priming("prompt", "", vec_index=None, repo=None,
                            conn=None, cfg=cfg, turn_features=features)


def test_no_features_never_gated(monkeypatch):
    out = _run(monkeypatch, _cfg(), [0.1, 0.05], None)
    assert out.gated == "full"
    assert len(out.semantic) == 2 and len(out.lexical) == 5


def test_floor_zero_disables_gate(monkeypatch):
    out = _run(monkeypatch, _cfg(turn_floor=0.0), [0.1],
               {"turn_idx": 50, "tool_density": 0.9, "notification": True})
    assert out.gated == "full" and out.semantic


def test_kickoff_unconditional_full(monkeypatch):
    out = _run(monkeypatch, _cfg(), [0.1],
               {"turn_idx": 2, "tool_density": 0.9, "notification": True})
    assert out.gated == "full" and out.semantic and len(out.lexical) == 5


def test_weak_turn_whispers_floor(monkeypatch):
    cap = {}
    out = _run(monkeypatch, _cfg(), [0.30, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01],
               {"turn_idx": 10, "tool_density": 0.0}, capture=cap)
    assert out.gated == "whisper-floor"
    assert len(out.semantic) == 5          # top whisper_k, NOT empty
    assert len(out.lexical) == 3           # lexical stays, thinner
    assert cap["lex_limit"] == 3


def test_dense_turn_whispers_regime(monkeypatch):
    out = _run(monkeypatch, _cfg(), [0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.44],
               {"turn_idx": 10, "tool_density": 0.75})
    assert out.gated == "whisper-regime"
    assert [sr.record.id for sr in out.semantic] == [
        f"rec_{i:08d}" for i in range(5)]
    assert len(out.lexical) == 3


def test_notification_whispers_even_on_strong_top(monkeypatch):
    out = _run(monkeypatch, _cfg(), [0.9] * 8,
               {"turn_idx": 10, "tool_density": 0.0, "notification": True})
    assert out.gated == "whisper-notification"
    assert len(out.semantic) == 5 and len(out.lexical) == 3


def test_normal_turn_full(monkeypatch):
    out = _run(monkeypatch, _cfg(), [0.9, 0.5],
               {"turn_idx": 10, "tool_density": 0.2})
    assert out.gated == "full"
    assert len(out.semantic) == 2 and len(out.lexical) == 5


def test_priming_result_default_gated():
    assert PrimingResult(semantic=[], lexical=[]).gated == "full"


def test_empty_semantic_whispers_floor(monkeypatch):
    """An EMPTY semantic bucket (dedup drained it) is the
    weakest field there is — it must whisper-floor, not slip through as full."""
    cap = {}
    out = _run(monkeypatch, _cfg(), [],
               {"turn_idx": 10, "tool_density": 0.0}, capture=cap)
    assert out.gated == "whisper-floor"
    assert out.semantic == []
    assert len(out.lexical) == 3 and cap["lex_limit"] == 3


def test_empty_semantic_dense_turn_still_floor(monkeypatch):
    # floor outranks regime in the trigger order — empty bucket wins.
    out = _run(monkeypatch, _cfg(), [],
               {"turn_idx": 10, "tool_density": 0.9})
    assert out.gated == "whisper-floor"


def test_turn_idx_none_is_kickoff_exempt(monkeypatch):
    """Unknown turn_idx fails OPEN — a genuine turn 1-3
    whose echo history could not be read keeps its unconditional full."""
    out = _run(monkeypatch, _cfg(), [0.1],
               {"turn_idx": None, "tool_density": 0.9, "notification": True})
    assert out.gated == "full"
    assert out.semantic and len(out.lexical) == 5


def test_weak_field_marker_wording():
    # "weak suggestions", not "ambient" (operationally unclear) and not
    # "not facts" (records ARE facts, valid at least at recording time).
    assert "weak suggestions" in _WEAK_FIELD_MARKER
    assert "ambient" not in _WEAK_FIELD_MARKER
    assert "not facts" not in _WEAK_FIELD_MARKER


def test_notification_detection():
    assert _is_notification_turn("<task-notification> ceva")
    assert _is_notification_turn("  <task-notification>x")
    assert not _is_notification_turn("prompt normal")
    assert not _is_notification_turn("")


def test_tool_density_from_transcript(tmp_path):
    lines = []
    for _ in range(2):
        lines.append({"type": "user", "message": {"content": "intrebare"}})
        lines.append({"type": "assistant",
                      "message": {"content": [{"type": "text", "text": "r"}]}})
    for _ in range(6):
        lines.append({"type": "assistant",
                      "message": {"content": [{"type": "tool_use", "id": "t",
                                               "name": "Bash", "input": {}}]}})
    p = tmp_path / "s.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
    assert _recent_tool_density(str(p)) == 0.6


def test_tool_density_missing_transcript(tmp_path):
    assert _recent_tool_density(None) is None
    assert _recent_tool_density(str(tmp_path / "absent.jsonl")) is None
