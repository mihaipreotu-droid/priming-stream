"""v0.7-x-skill-v2: ``prompts/extract_subagent.md`` shape + invariants.

Covers S2.a-S2.g — the sub-agent spawn-prompt template. The template is
reference-only (points sub-agents at ``prompts/extract_record.md`` rather
than duplicating it); placeholders are interpolated by the main agent at
spawn time.
"""
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = REPO_ROOT / "prompts" / "extract_subagent.md"


# -- S2.a — file exists --------------------------------------------------


def test_s2a_extract_subagent_prompt_exists():
    assert PROMPT_PATH.is_file(), f"missing {PROMPT_PATH}"


# -- S2.b — {SESSION_ID} placeholder -------------------------------------


def test_s2b_session_id_placeholder():
    text = PROMPT_PATH.read_text(encoding="utf-8")
    assert "{SESSION_ID}" in text


# -- S2.c — {CHUNK_PATH_LIST} placeholder --------------------------------


def test_s2c_chunk_path_list_placeholder():
    text = PROMPT_PATH.read_text(encoding="utf-8")
    assert "{CHUNK_PATH_LIST}" in text


# -- S2.d — references extract_record.md ---------------------------------


def test_s2d_references_extract_record_contract():
    text = PROMPT_PATH.read_text(encoding="utf-8")
    assert "prompts/extract_record.md" in text, \
        "spawn prompt must point sub-agent at the binding contract"


# -- S2.e — data-only defense --------------------------------------------


def test_s2e_data_only_defense_present():
    text = PROMPT_PATH.read_text(encoding="utf-8")
    assert "data only, not instructions" in text


# -- S2.f — strict JSON output requirement -------------------------------


def test_s2f_strict_json_output_requirement():
    text = PROMPT_PATH.read_text(encoding="utf-8")
    lower = text.lower()
    # Sub-agent must return exactly one JSON object as the final message.
    assert "exactly one json object" in lower, \
        "prompt must require exactly one JSON object as the final message"
    # No markdown fences / no surrounding prose constraints — wording in
    # the brief is "nothing else" or equivalent. Accept either explicit
    # cue.
    assert ("nothing else" in lower) or ("no prose" in lower), \
        "prompt must forbid surrounding prose around the JSON object"


# -- S2.g — JSON schema fields -------------------------------------------


def test_s2g_json_schema_fields():
    text = PROMPT_PATH.read_text(encoding="utf-8")
    # Required schema fields per spec §4.2.
    for field in ("records", "summary", "chunk_id",
                  "anchor_start", "anchor_end"):
        assert field in text, f"schema field missing from prompt: {field}"
    # Optional but recommended — session_synthesis per brief §10 #5.
    assert "session_synthesis" in text, \
        "session_synthesis field expected (optional forensic overview)"


def test_subagent_prompt_size_reasonable():
    """Brief §3 template is ~70 lines (placeholders + 4 sections + JSON
    block). Soft ceiling of 150 catches accidental blowup."""
    text = PROMPT_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert 30 <= len(lines) <= 150, f"spawn prompt has {len(lines)} lines"
