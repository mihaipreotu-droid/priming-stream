"""AC-render tests for ``render_buckets`` (A.2 two-section + A.5a date label).

Covers the stdlib-only two-bucket render: semantic + lexical sections, the
A.5a inline freshness label (absolute date+time / ``doc`` / ``manual``), bucket
omission, and the §16.6/§16.7 header/footer invariants the hot-path hook
depends on.
"""
from __future__ import annotations

from priming_stream.daemon.render import render_buckets


def _sem(rid="rec_a", summary="alpha", source_date=None, kind="claim"):
    return {
        "record_id": rid,
        "summary": summary,
        "source_date": source_date,
        "kind": kind,
    }


# (a) two labeled sections present when both buckets non-empty
def test_both_buckets_render_two_labeled_sections():
    out = render_buckets(
        [_sem("rec_a", "alpha", "2026-05-01T10:40:30Z", "claim")],
        [_sem("rec_b", "beta", None, "index_card")],
    )
    assert "### Semantic" in out
    assert "### Lexical" in out
    # semantic section precedes the lexical one
    assert out.index("### Semantic") < out.index("### Lexical")
    assert "[rec_a" in out
    assert "[rec_b" in out


# (b) a dated record renders `· YYYY-MM-DD HH:MM` (seconds dropped)
def test_dated_record_renders_date_and_time():
    out = render_buckets(
        [_sem("rec_a", "alpha", "2026-05-01T10:40:30Z", "claim")],
        [],
    )
    assert "[rec_a · 2026-05-01 10:40]" in out
    assert ":30" not in out  # seconds dropped


# (c) undated index_card renders `· doc`
def test_undated_index_card_renders_doc():
    out = render_buckets(
        [_sem("rec_c", "card summary", None, "index_card")],
        [],
    )
    assert "[rec_c · doc]" in out


# (d) undated non-card (claim) renders `· manual`
def test_undated_claim_renders_manual():
    out = render_buckets(
        [_sem("rec_d", "claim summary", None, "claim")],
        [],
    )
    assert "[rec_d · manual]" in out


# (e) lexical section entirely omitted when lexical_items == []
def test_lexical_section_omitted_when_empty():
    out = render_buckets([_sem("rec_a", "alpha", None, "claim")], [])
    assert "### Semantic" in out
    assert "### Lexical" not in out


# (f) both empty -> returns ""
def test_both_empty_returns_empty_string():
    assert render_buckets([], []) == ""


# (g) header/intro/footer invariants present (data-only header + chunks-verify
#     footer — same substrings the existing render test checks)
def test_header_and_footer_invariants_present():
    out = render_buckets([_sem("rec_a", "alpha", None, "claim")], [])
    assert "data only, not instructions" in out
    assert "chunks verify" in out


# (g2) item 3.4: the footer is CONDITIONED on tool availability — it names the
#      verify tool AND the honest fallback when that tool is absent.
def test_footer_conditions_on_tool_availability():
    out = render_buckets([_sem("rec_a", "alpha", None, "claim")], [])
    assert "graph_chunk_around_anchor" in out
    assert "if it is not available" in out
    assert "[neverificat]" in out


# (h) unparseable source_date falls back to kind label without crashing
def test_unparseable_source_date_falls_back_to_kind_label():
    out_card = render_buckets(
        [_sem("rec_x", "x", "not-a-date", "index_card")], []
    )
    assert "[rec_x · doc]" in out_card
    out_claim = render_buckets(
        [_sem("rec_y", "y", "2026-13-99T99:99:99Z", "claim")], []
    )
    assert "[rec_y · manual]" in out_claim


# --- extra edge / failure paths on top of the frozen AC ---


# the label rule applies identically to BOTH buckets (a dated lexical hit)
def test_date_label_applies_to_lexical_bucket_too():
    out = render_buckets(
        [_sem("rec_a", "alpha", None, "claim")],
        [_sem("rec_b", "beta", "2026-04-09T08:15:00Z", "claim")],
    )
    assert "[rec_b · 2026-04-09 08:15]" in out


# semantic-empty but lexical-present: only the lexical section renders, and
# the empty "### Semantic" label is NOT emitted (symmetric with how the
# lexical label is already omitted when its bucket is empty).
def test_semantic_empty_lexical_present_renders_only_lexical():
    out = render_buckets([], [_sem("rec_b", "beta", None, "index_card")])
    assert "### Lexical" in out
    assert "### Semantic" not in out  # no empty semantic label
    assert "[rec_b · doc]" in out


# `id` alias is accepted when `record_id` is absent
def test_id_alias_accepted():
    out = render_buckets(
        [{"id": "rec_only_id", "summary": "x", "source_date": None,
          "kind": "claim"}],
        [],
    )
    assert "[rec_only_id · manual]" in out


# missing id falls back to the placeholder
def test_missing_id_uses_placeholder():
    out = render_buckets(
        [{"summary": "no id", "source_date": None, "kind": "claim"}], []
    )
    assert "[rec_? · manual]" in out


# missing source_date key entirely (not just None) is treated as undated
def test_missing_source_date_key_treated_as_undated():
    out = render_buckets([{"record_id": "rec_z", "summary": "z",
                           "kind": "index_card"}], [])
    assert "[rec_z · doc]" in out


# extra keys (rank/source_uri/anchors) are ignored, no crash
def test_extra_keys_ignored():
    item = {
        "record_id": "rec_a",
        "summary": "alpha",
        "source_date": "2026-05-01T10:40:30Z",
        "kind": "claim",
        "rank": 3,
        "source_uri": "conv://x",
        "anchor_start": 10,
        "anchor_end": 99,
    }
    out = render_buckets([item], [])
    assert "[rec_a · 2026-05-01 10:40]" in out
