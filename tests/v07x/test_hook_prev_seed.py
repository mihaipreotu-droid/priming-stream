"""P5 response-seed — unit tests for the hook's transcript-tail prev reader.

Covers ``_last_assistant_text`` (transcript parsing: last assistant text,
sidechain skip, tool_use-only skip, string-content tolerance, corrupt lines,
missing file) and ``_slice_prev`` (tail-only default, head+tail shape,
short passthrough). The slice constants were decided empirically —
these tests pin the mechanics, not the constants' values.
"""
from __future__ import annotations

import json

from priming_stream.hooks.user_prompt_submit import (
    _PREV_HEAD_CHARS,
    _PREV_TAIL_CHARS,
    _last_assistant_text,
    _slice_prev,
)


def _asst(text_blocks, sidechain=False, ts="2026-07-21T10:00:00.000Z"):
    content = [{"type": "text", "text": t} for t in text_blocks]
    content.append({"type": "tool_use", "id": "tu_1", "name": "Read", "input": {}})
    return {"type": "assistant", "isSidechain": sidechain, "timestamp": ts,
            "message": {"role": "assistant", "content": content}}


def _user(text="hello"):
    return {"type": "user", "message": {"role": "user", "content": text}}


def _write(tmp_path, lines):
    p = tmp_path / "session.jsonl"
    p.write_text("\n".join(json.dumps(ln, ensure_ascii=False) for ln in lines) + "\n",
                 encoding="utf-8")
    return str(p)


def test_last_assistant_text_returns_newest(tmp_path):
    path = _write(tmp_path, [
        _user(), _asst(["prima replică"]), _user(), _asst(["a doua replică"]),
    ])
    assert _last_assistant_text(path) == "a doua replică"


def test_sidechain_lines_skipped(tmp_path):
    path = _write(tmp_path, [
        _asst(["replica principală"]),
        _asst(["zgomot de subagent"], sidechain=True),
    ])
    assert _last_assistant_text(path) == "replica principală"


def test_tool_use_only_turn_skipped(tmp_path):
    tool_only = {"type": "assistant",
                 "message": {"role": "assistant",
                             "content": [{"type": "tool_use", "id": "tu",
                                          "name": "Bash", "input": {}}]}}
    path = _write(tmp_path, [_asst(["textul real"]), tool_only])
    assert _last_assistant_text(path) == "textul real"


def test_multiple_text_blocks_joined(tmp_path):
    path = _write(tmp_path, [_asst(["bloc unu", "bloc doi"])])
    out = _last_assistant_text(path)
    assert "bloc unu" in out and "bloc doi" in out


def test_string_content_tolerated(tmp_path):
    path = _write(tmp_path, [
        {"type": "assistant", "message": {"role": "assistant",
                                          "content": "conținut simplu"}},
    ])
    assert _last_assistant_text(path) == "conținut simplu"


def test_corrupt_lines_skipped(tmp_path):
    p = tmp_path / "session.jsonl"
    p.write_text(
        json.dumps(_asst(["valid"])) + "\n{broken json\n\n", encoding="utf-8",
    )
    assert _last_assistant_text(str(p)) == "valid"


def test_missing_or_empty_path_returns_empty(tmp_path):
    assert _last_assistant_text(None) == ""
    assert _last_assistant_text("") == ""
    assert _last_assistant_text(str(tmp_path / "absent.jsonl")) == ""


def test_slice_short_passthrough():
    assert _slice_prev("scurt") == "scurt"
    exact = "x" * (_PREV_HEAD_CHARS + _PREV_TAIL_CHARS)
    assert _slice_prev(exact) == exact


def test_slice_default_tail_only():
    """Empirical winner (21-07): tail-only — the slice must END with the
    reply's tail and carry no head fragment when _PREV_HEAD_CHARS == 0."""
    text = "A" * 3000 + "FINAL"
    out = _slice_prev(text)
    assert out == text[-(_PREV_HEAD_CHARS + _PREV_TAIL_CHARS):] or (
        _PREV_HEAD_CHARS > 0  # guard: if constants recalibrate, shape test below covers
    )
    assert out.endswith("FINAL")
    assert len(out) <= _PREV_HEAD_CHARS + _PREV_TAIL_CHARS + 10


def test_slice_head_tail_shape_reachable():
    text = "HEADSTART" + "x" * 3000 + "TAILEND"
    out = _slice_prev(text, head=100, tail=200)
    assert out.startswith("HEADSTART")
    assert out.endswith("TAILEND")
    assert "…" in out
