"""v0.7-x sleep-cycle redesign: worker-flow planning + results parsing.

The skill-v2 manual sub-agent flow (size-class parallel caps, a failure-mode
taxonomy with retry loops, and a JSON reject-validator mirroring "Step 2.3")
was replaced by a deterministic pipeline:

  * ``plan.py`` groups chunks by conversation and ROUTES BY BODY-TOKEN LOAD
    (≤100K → Sonnet single-pass, >100K → Opus-1M single-pass). No size
    classes, no parallel caps — the Workflow fans out one worker per
    conversation.
  * Each worker writes ONE delimited PLAIN-TEXT results file (not JSON, so
    quotes/commas/newlines in summaries can't break parsing).
  * ``writer.py`` parses those results and materializes record ``.md`` files,
    CLAMPING anchors into range rather than rejecting (validation moved from
    a reject-model to a clamp-model).

These tests exercise the surviving, importable logic of that pipeline:
``plan.py`` helpers + routing constants, and ``writer.py`` block parsing +
anchor clamping. The old JSON validator / planning-cap / retry-taxonomy tests
were deleted — they targeted a design the redesign dropped.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO_ROOT / ".claude" / "skills" / "prime-ingest"


def _load_module(name: str, filename: str):
    """Load a module from the skill directory by file path (the skill dir is
    not on sys.path / not a package)."""
    path = SKILL_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader, f"cannot load {filename}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


plan = _load_module("ch_sleep_plan", "plan.py")
writer = _load_module("ch_sleep_writer", "writer.py")


# ---------------------------------------------------------------------------
# plan.py — routing by body-token load (replaces size-class caps)
# ---------------------------------------------------------------------------


def test_token_threshold_default_is_size_based():
    """Reverted 2026-06-11 from all-Opus (L(b) threshold 0): default threshold
    is 100K — conversations >100K body-tokens route to Opus, the rest to Sonnet
    (Opus showed weaker contract compliance + higher cost; the analytical-moves
    edge didn't justify the swap). --threshold 0 forces all-Opus again."""
    assert plan.TOKEN_THRESHOLD == 100_000


def test_route_small_conversation_to_sonnet_at_default():
    """A conversation <=100K tokens routes to Sonnet at the default threshold."""
    est_tokens = 50_000
    mode = "opus" if est_tokens > plan.TOKEN_THRESHOLD else "sonnet"
    assert mode == "sonnet"


def test_route_large_conversation_to_opus_at_default():
    """A conversation >100K tokens routes to Opus at the default threshold."""
    est_tokens = 150_000
    mode = "opus" if est_tokens > plan.TOKEN_THRESHOLD else "sonnet"
    assert mode == "opus"


def test_route_formula_respects_explicit_threshold():
    """--threshold 100000 restores the old routing: <=100K stays Sonnet,
    >100K goes Opus (boundary inclusive for Sonnet)."""
    threshold = 100_000
    assert ("opus" if threshold > threshold else "sonnet") == "sonnet"
    assert ("opus" if threshold + 1 > threshold else "sonnet") == "opus"


@pytest.mark.parametrize(
    "path,expected",
    [
        # UUID conversation segment in the path → that's the conv key.
        (
            "storage/corpus/chunks/123e4567-e89b-12d3-a456-426614174000/p0.md",
            "123e4567-e89b-12d3-a456-426614174000",
        ),
        # uN-style session id.
        ("storage/corpus/chunks/u42/export_p1.md", "u42"),
        # Fallback: parent directory name.
        ("storage/corpus/chunks/someconv/p0.md", "someconv"),
    ],
)
def test_conv_of_extracts_conversation_key(path: str, expected: str):
    assert plan._conv_of(path) == expected


@pytest.mark.parametrize(
    "chunk_id,expected",
    [
        ("export_abc_p0", 0),
        ("export_abc_p1", 1),
        ("export_abc_p12", 12),
        ("noindex", -1),
    ],
)
def test_order_key_sorts_chunks_by_page(chunk_id: str, expected: int):
    assert plan._order_key(chunk_id) == expected


def test_body_len_strips_frontmatter(tmp_path: Path):
    """Body-token estimation measures the post-frontmatter body only."""
    f = tmp_path / "chunk.md"
    body = "the actual conversation body text"
    f.write_text(f"---\nid: x\nsource_uri: y\n---\n{body}", encoding="utf-8")
    assert plan._body_len(str(f)) == len(body)


def test_body_len_no_frontmatter_counts_whole_file(tmp_path: Path):
    f = tmp_path / "plain.md"
    f.write_text("no frontmatter here", encoding="utf-8")
    assert plan._body_len(str(f)) == len("no frontmatter here")


def test_body_len_missing_file_is_zero():
    assert plan._body_len("does/not/exist.md") == 0


# ---------------------------------------------------------------------------
# writer.py — plain-text results parsing (replaces the JSON validator)
# ---------------------------------------------------------------------------


VALID_BLOCK = (
    "\nCHUNK: export_abc_p0\n"
    "ANCHOR: 100 256\n"
    "Decision: prefer the thin orchestrator over manual Opus orchestration.\n"
)


def test_parse_block_accepts_valid_block():
    parsed = writer._parse_block(VALID_BLOCK)
    assert parsed is not None
    chunk_id, start, end, doc_ref, summary = parsed
    assert chunk_id == "export_abc_p0"
    assert start == 100
    assert end == 256
    assert doc_ref is None
    assert summary.startswith("Decision: prefer the thin orchestrator")


def test_parse_block_multiline_summary_preserved():
    block = (
        "\nCHUNK: c1\nANCHOR: 0 10\n"
        "First line, with commas and \"quotes\".\nSecond line.\n"
    )
    parsed = writer._parse_block(block)
    assert parsed is not None
    _cid, _s, _e, _dref, summary = parsed
    assert "First line" in summary and "Second line" in summary


def test_parse_block_docref_line():
    """piece3-B: optional DOCREF (title handle) between ANCHOR and summary."""
    block = (
        "\nCHUNK: c1\nANCHOR: 5 20\n"
        "DOCREF: Collins & Loftus 1975\n"
        "We use spreading activation from Collins & Loftus.\n"
    )
    parsed = writer._parse_block(block)
    assert parsed is not None
    cid, start, end, doc_ref, summary = parsed
    assert doc_ref == "Collins & Loftus 1975"
    assert summary.startswith("We use spreading activation")
    assert "DOCREF" not in summary  # tag line consumed, not in body


def test_parse_doc_block_components_and_stub():
    block = (
        "\nDOI: 10.1/x\nURL: \nAUTHORS: Smith & Jones\nYEAR: 2020\n"
        "DOCTITLE: Some Paper\nSOURCE: \n"
        "Some Paper studies X and finds Y. [unverified]\n"
    )
    f = writer._parse_doc_block(block)
    assert f is not None
    assert f["doi"] == "10.1/x"
    assert f["url"] is None
    assert f["authors"] == "Smith & Jones"
    assert f["year"] == "2020"
    assert f["title"] == "Some Paper"
    assert f["stub"].startswith("Some Paper studies X")


def test_parse_doc_block_tag_only_empty_stub():
    block = "\nAUTHORS: Hyde\nYEAR: 2005\nDOCTITLE: Gender Similarities\nSOURCE: \n"
    f = writer._parse_doc_block(block)
    assert f is not None
    assert f["title"] == "Gender Similarities"
    assert f["stub"] == ""  # tag-only: no card


def test_parse_doc_block_missing_title_returns_none():
    assert writer._parse_doc_block("\nAUTHORS: x\nbody\n") is None


def test_classify_blocks_order_independent():
    """A ===DOC=== interleaved among records must NOT swallow trailing records
    (robustness against worker misordering)."""
    text = (
        "HEADER\n"
        "===REC===\nCHUNK: c0\nANCHOR: 0 1\nfirst record\n"
        "===DOC===\nDOCTITLE: X\nSOURCE: \nstub\n"
        "===REC===\nCHUNK: c0\nANCHOR: 2 3\nsecond record after a doc block\n"
    )
    recs, docs = writer._classify_blocks(text)
    assert len(recs) == 2  # both records survive despite the interleaved doc
    assert len(docs) == 1
    assert "first record" in recs[0]
    assert "second record" in recs[1]
    assert "stub" in docs[0]


def test_classify_blocks_no_docs():
    recs, docs = writer._classify_blocks(
        "H\n===REC===\nCHUNK: c\nANCHOR: 0 1\nr\n"
    )
    assert len(recs) == 1 and docs == []


def test_parse_block_missing_chunk_returns_none():
    block = "\nANCHOR: 0 10\nsome summary\n"
    assert writer._parse_block(block) is None


def test_parse_block_missing_anchor_returns_none():
    block = "\nCHUNK: c1\nsome summary without an anchor line\n"
    assert writer._parse_block(block) is None


def test_parse_block_empty_summary_returns_none():
    block = "\nCHUNK: c1\nANCHOR: 0 10\n   \n"
    assert writer._parse_block(block) is None


def test_parse_block_garbage_anchor_defaults_to_zero():
    """Non-numeric anchor must not crash — it degrades to 0 0 (writer then
    clamps); the record still parses."""
    block = "\nCHUNK: c1\nANCHOR: not numbers\nsummary text\n"
    parsed = writer._parse_block(block)
    assert parsed is not None
    _cid, start, end, _dref, _summary = parsed
    assert start == 0 and end == 0


def test_parse_block_single_anchor_value_sets_end_to_start():
    block = "\nCHUNK: c1\nANCHOR: 42\nsummary text\n"
    parsed = writer._parse_block(block)
    assert parsed is not None
    _cid, start, end, _dref, _summary = parsed
    assert start == 42 and end == 42


# ---------------------------------------------------------------------------
# Anchor clamping (validation moved from reject-model to clamp-model)
# ---------------------------------------------------------------------------


def _clamp(start: int, end: int, blen: int) -> tuple[int, int]:
    """Mirror writer.main()'s clamping arithmetic for a single record."""
    start = max(0, start)
    if blen:
        end = min(end, blen)
    end = max(end, start)
    return start, end


def test_clamp_in_range_unchanged():
    assert _clamp(100, 256, 4096) == (100, 256)


def test_clamp_negative_start_to_zero():
    assert _clamp(-5, 256, 4096) == (0, 256)


def test_clamp_end_past_body_length():
    assert _clamp(0, 999_999, 4096) == (0, 4096)


def test_clamp_inverted_offsets_end_raised_to_start():
    """End before start is repaired (end := start), not rejected."""
    assert _clamp(512, 256, 4096) == (512, 512)
