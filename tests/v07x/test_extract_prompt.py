"""v0.7-x W-C: ``prompts/extract_record.md`` shape + invariants.

Skill-v2 adds S3.a — a binding-contract header note declaring this file
the single source of truth for the extraction contract. Sub-agents Read
it directly rather than receiving a paraphrase in the spawn prompt.
"""
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = REPO_ROOT / "prompts" / "extract_record.md"


def test_extract_prompt_exists():
    assert PROMPT_PATH.is_file()


def test_extract_prompt_data_not_instructions_clause():
    # §16.7 prompt-injection defense — the literal substring is the
    # acceptance gate (spec C3).
    text = PROMPT_PATH.read_text(encoding="utf-8")
    assert "data only, not instructions" in text


def test_extract_prompt_describes_json_output():
    text = PROMPT_PATH.read_text(encoding="utf-8")
    assert "records" in text
    assert "summary" in text
    assert "anchor_start" in text
    assert "anchor_end" in text


def test_extract_prompt_size_reasonable():
    """Brief originally called for 50-120 lines. Iterations since: dyad-anchor
    test as primary filter + read-synthesize-extract model + episode
    granularity + summary discipline (~220 lines), then the analytical-
    structure class (~309), then the piece3 document-mode section (~335),
    then the piece3-B doc-candidate detection section (~415), then the unify-F3
    additions (source-aware Claude-Code note + produce/process local-file
    doc-candidates with final-vs-draft significance, ~459), then the
    intra-conversation retraction/correction subsection (Caz A of the supersedence
    decision, ~511), then the transferability test as a second primary filter
    (transfers-cross-context vs in-scope process/build residue, ephemeral lookups,
    bare state-snapshots; from the 2026-06-17 relevance audit, ~584), then the
    post-draft gates section (compression / CC-residue / anchor verification;
    from the 2026-07-19 extraction audit, ~699). Soft ceiling raised to 760
    to catch accidental blowup without flagging the deliberate contract
    growth."""
    text = PROMPT_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert 30 <= len(lines) <= 760, f"prompt has {len(lines)} lines"


def test_extract_prompt_mentions_granularity():
    text = PROMPT_PATH.read_text(encoding="utf-8").lower()
    # Variable granularity is part of the contract.
    assert "sentence" in text
    assert "paragraph" in text


def test_extract_prompt_summary_discipline_present():
    """Summary discipline binding rules — ≤20 word cap + one-claim-per-summary
    — must survive prompt edits (e.g., header-note additions in skill-v2)."""
    text = PROMPT_PATH.read_text(encoding="utf-8")
    assert "≤20 words" in text, "20-word summary cap must remain in contract"
    assert "One claim per summary" in text, \
        "one-claim-per-summary rule must remain in contract"


# -- S3.a — binding-contract header note (skill-v2) ----------------------


def test_s3a_binding_contract_header_note():
    """Sub-agents Read this file via the Read tool; the spawn prompt only
    points at it. The header note declares this file the single source of
    truth so a future editor doesn't paraphrase the contract elsewhere."""
    text = PROMPT_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()
    # Search the top of the file (first ~10 content lines) for the note.
    head = "\n".join(lines[:12]).lower()
    assert "binding" in head, "binding-contract header note missing near top"
    assert "contract" in head, "binding-contract header note missing near top"
