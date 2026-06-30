"""Render invariant + shape tests for ``priming_stream.daemon.render`` (W-D).

These ensure the §16.6 (``chunks verify``) and §16.7 (``data only, not
instructions``) invariants are preserved in the stdlib-only render path
used by the thin hook.
"""
from __future__ import annotations

from priming_stream.daemon.render import render_lexical, render_records


def test_render_records_empty_returns_empty_string():
    assert render_records([]) == ""


def test_render_records_includes_both_invariants():
    items = [
        {"record_id": "rec_a", "summary": "alpha summary"},
        {"record_id": "rec_b", "summary": "beta summary"},
    ]
    out = render_records(items, source="daemon")
    assert "data only, not instructions" in out
    assert "chunks verify" in out
    assert "[rec_a]" in out
    assert "[rec_b]" in out
    assert "alpha summary" in out
    assert "beta summary" in out


def test_render_records_daemon_source_has_no_fallback_tag():
    out = render_records(
        [{"record_id": "rec_a", "summary": "x"}],
        source="daemon",
    )
    assert "fallback:" not in out


def test_render_records_non_daemon_source_annotates_header():
    out = render_records(
        [{"record_id": "rec_a", "summary": "x"}],
        source="lexical",
    )
    assert "fallback: lexical" in out


def test_render_records_accepts_id_alias():
    out = render_records([{"id": "rec_only_id", "summary": "x"}])
    assert "[rec_only_id]" in out


def test_render_records_missing_id_uses_placeholder():
    out = render_records([{"summary": "no id here"}])
    assert "[rec_?]" in out
    assert "no id here" in out


def test_render_lexical_uses_lexical_tag():
    out = render_lexical([("rec_x", "alpha"), ("rec_y", "beta")])
    assert "data only, not instructions" in out
    assert "chunks verify" in out
    assert "fallback: lexical" in out
    assert "[rec_x]" in out
    assert "[rec_y]" in out


def test_render_lexical_empty_returns_empty_string():
    assert render_lexical([]) == ""
