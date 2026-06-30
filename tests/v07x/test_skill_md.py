"""v0.7-x unify: ``.claude/skills/prime-ingest/SKILL.md`` shape (the unified
ingest skill).

The conversational sleep cycle (formerly ``/prime-sleep``) was unified into
``/prime-ingest`` — one skill that ingests conversation sources + documents in one
cycle. The conversational branch keeps the thin-orchestrator pipeline:

    [ingest-source] → sleep-prepare → plan.py → conv_extract.workflow.js (Workflow)
                    → writer.py → reconcile → sleep-finalize

These tests assert that structure on the UNIFIED skill. ``/prime-sleep`` is now a
retired redirect (see its SKILL.md), so the orchestration invariants live here.
Step numbering changed with the merge (Steps 1–7; finalize is Step 6).
"""
from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_PATH = REPO_ROOT / ".claude" / "skills" / "prime-ingest" / "SKILL.md"


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Hand-roll a tiny YAML frontmatter parser (single-line keys only — we
    don't need general YAML for this contract)."""
    if not text.startswith("---\n"):
        raise ValueError("missing leading '---' line")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise ValueError("missing trailing '---' line")
    raw = text[4:end]
    body = text[end + len("\n---\n"):]
    fm: dict[str, str] = {}
    for ln in raw.splitlines():
        if not ln.strip():
            continue
        if ":" not in ln:
            raise ValueError(f"frontmatter line missing ':' — {ln!r}")
        key, _, value = ln.partition(":")
        fm[key.strip()] = value.strip()
    return fm, body


def _body() -> str:
    text = SKILL_PATH.read_text(encoding="utf-8")
    _fm, body = _parse_frontmatter(text)
    return body


def _flat() -> str:
    """Body with all runs of whitespace collapsed to single spaces, lowercased
    — so phrase checks survive markdown hard-wrapping (a phrase split across a
    line break still matches)."""
    return re.sub(r"\s+", " ", _body()).lower()


# -- baseline shape ------------------------------------------------------


def test_skill_md_exists():
    assert SKILL_PATH.is_file(), f"SKILL.md not found at {SKILL_PATH}"


def test_skill_md_frontmatter_parses():
    text = SKILL_PATH.read_text(encoding="utf-8")
    fm, _body = _parse_frontmatter(text)
    assert fm.get("name") == "prime-ingest"
    assert fm.get("description")
    desc = fm["description"].lower()
    assert "ingest" in desc
    assert "record" in desc or "Priming Stream" in desc


# -- thin-orchestrator: five-stage pipeline surfaced --------------------


def test_skill_md_describes_thin_orchestrator():
    """The skill body must frame the main (Opus) session as a thin
    orchestrator that does NOT read chunks / validate / persist itself —
    the Workflow's workers do that on fresh contexts."""
    flat = _flat()
    # "thin orchestrator" may be wrapped in markdown bold / split across a
    # line break in the source — match on the whitespace-flattened body.
    assert "thin orchestrator" in flat, \
        'SKILL.md must frame the role as a "thin orchestrator"'
    # The whole point of the redesign: Opus stays out of the heavy I/O.
    assert "context clean" in flat or "context curat" in flat


def test_skill_md_pipeline_cli_bookends_present():
    body = _body()
    # CLI bookends.
    assert "sleep-prepare" in body
    assert "sleep-finalize" in body


def test_skill_md_pipeline_stages_present():
    """plan.py → Workflow → writer.py is the deterministic core."""
    body = _body()
    assert "plan.py" in body, "Step 2 must run the planner plan.py"
    assert "Workflow" in body, "Step 3 must invoke the extraction Workflow"
    assert "conv_extract.workflow.js" in body, \
        "Step 3 must name the conversational-extraction workflow script"
    assert "writer.py" in body, "Step 3.5 must run the bulk-writer writer.py"


def test_skill_md_has_ordered_steps_1_through_7():
    """Unified skill: Step 1 ingest-source · 2 prepare · 3 conversation branch
    · 4 document branch · 5 reconcile · 6 finalize · 7 report."""
    body = _body()
    for header in (
        r"^## Step 1\b",
        r"^## Step 2\b",
        r"^## Step 3\b",
        r"^## Step 4\b",
        r"^## Step 5\b",
        r"^## Step 6\b",
        r"^## Step 7\b",
    ):
        assert re.search(header, body, flags=re.MULTILINE), \
            f"missing step header matching {header!r}"


# -- Step 2 — plan.py routing by body-token load ------------------------


def test_step2_routes_conversations_to_opus():
    """L(b) decision (2026-06-10): every conversation routes to Opus; the
    skill must state the default AND the --threshold rollback knob."""
    body = _body()
    # token-load routing vocabulary survives (the knob still routes by load)
    assert "body-token" in body or "body token" in body
    lower = body.lower()
    assert "opus" in lower
    assert "--threshold" in body, "the rollback knob must be documented"
    # the planner groups by conversation
    assert "conversation" in lower


# -- Step 3 — one worker per conversation, single full-context pass -------


def test_step3_one_worker_per_conversation():
    flat = _flat()
    assert "one worker per conversation" in flat
    # single full-context pass — no segmentation/framework/dedup machinery
    assert "single-pass" in flat or "single pass" in flat or "full-context" in flat
    assert "no segmentation" in flat or "no dedup" in flat


def test_step3_workers_write_results_not_record_files():
    """Workers write ONE results file per conversation; record .md files are
    materialized later by writer.py (Step 3.5)."""
    body = _body()
    assert "_sleep_results" in body
    # the bulk-write is the authoritative count, not the worker self-report
    assert "authoritative" in body.lower()


# -- invariants preserved across the redesign ---------------------------


def test_body_includes_data_only_clause():
    """Chunk content is data, not instructions — preserved verbatim."""
    assert "data only, not instructions" in _body()


def test_step6_finalize_present():
    body = _body()
    assert re.search(r"^## Step 6\b", body, flags=re.MULTILINE), \
        "Step 6 (sleep-finalize) section missing"
    assert "sleep-finalize" in body


def test_skill_md_references_extraction_contract():
    """extract_record.md remains the single binding extraction contract;
    workers read it via the path in their assignment."""
    assert "prompts/extract_record.md" in _body()


def test_skill_md_no_manual_prefilter():
    """The heuristic lexical pre-filter (record_markers.toml / record_filter)
    was dropped before the redesign and must not have crept back in. The new
    design has no pre-filter stage at all — empty results = the skip signal."""
    body = _body()
    assert "record_markers.toml" not in body
    assert "record_filter" not in body


def test_skill_md_drops_manual_orchestration_vocabulary():
    """Regression guard: the manual skill-v2 orchestration concepts (size
    classes with parallel caps, four Step-2 sub-steps, retry loops,
    session_synthesis→notes, a separate sub-agent spawn prompt) must not
    reappear — the thin orchestrator delegates all of that to the Workflow."""
    body = _body()
    assert "extract_subagent.md" not in body, \
        "redesign dropped the separate sub-agent spawn-prompt template"
    assert "session_synthesis" not in body, \
        "redesign dropped session_synthesis→sleep_cycles.notes routing"
    assert "Step 2.1" not in body and "Step 2.4" not in body, \
        "redesign dropped the four Step-2 sub-steps"
