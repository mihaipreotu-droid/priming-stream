"""UserPromptSubmit hook end-to-end (spec §D6-D9, §D12).

The hook is the v0.7-x-bridge-daemon thin shape: stdlib + daemon.client +
daemon.fallback_lexical + daemon.render + core.{config,paths}. No
fastembed / chromadb / priming_stream.bridge.

These tests patch ``daemon.client.spread`` and the lexical search to drive
the three tiers (warm daemon / fallback / empty) plus the never-crash
discipline. The subprocess-based D12 test invokes the hook as a real
process and asserts stdout is exactly one JSON object.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from priming_stream.daemon import client as daemon_client
from priming_stream.daemon import fallback_lexical
from priming_stream.hooks import user_prompt_submit as hook


# ----------------------------------------------------------------- helpers


def _drive_hook(monkeypatch, capsys, event: dict) -> str:
    """Feed ``event`` as JSON on stdin; return whatever the hook wrote
    to stdout. Captures via pytest's capsys so we also see prints (any
    print would be a hook-contract violation)."""
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(json.dumps(event, ensure_ascii=False)),
    )
    hook.main()
    return capsys.readouterr().out


# --------------------------------------------------------- D7 — warm daemon


def test_d7_warm_daemon_returns_rendered_buckets(monkeypatch, capsys):
    """When the daemon returns the two buckets, the hook renders both
    sections via ``render_buckets`` + emits them (Component A shape)."""

    def fake_spread(prompt, prev="", *, session_id=None,
                    deadline_ms=800, connect_timeout_ms=100):
        return {
            "semantic": [
                {"record_id": "rec_001", "summary": "alpha summary",
                 "rank": 1, "source_uri": "qmd://x",
                 "anchor_start": 0, "anchor_end": 0,
                 "source_date": "2026-05-01T10:40:00Z", "kind": "claim"},
                {"record_id": "rec_002", "summary": "beta summary",
                 "rank": 2, "source_uri": "qmd://y",
                 "anchor_start": 0, "anchor_end": 0,
                 "source_date": None, "kind": "claim"},
            ],
            "lexical": [
                {"record_id": "rec_card", "summary": "a cited paper",
                 "rank": 1, "source_uri": "file:///p.pdf",
                 "anchor_start": 0, "anchor_end": 0,
                 "source_date": None, "kind": "index_card"},
            ],
            "spread_ms": 12.3,
            "daemon_version": "v0.7-x-bridge-daemon",
        }

    monkeypatch.setattr(daemon_client, "spread", fake_spread)

    out = _drive_hook(monkeypatch, capsys,
                      {"prompt": "tell me", "session_id": "s1"})
    payload = json.loads(out)
    text = payload["hookSpecificOutput"]["additionalContext"]
    assert "data only, not instructions" in text
    assert "chunks verify" in text
    # Two-section render with A.5a freshness labels.
    assert "Semantic (associative spread)" in text
    assert "Lexical (term / citation match)" in text
    assert "[rec_001 · 2026-05-01 10:40]" in text
    assert "[rec_002 · manual]" in text   # undated claim
    assert "[rec_card · doc]" in text     # undated index_card
    # Daemon source → no fallback annotation in header.
    assert "fallback:" not in text


def test_d7_warm_daemon_semantic_only_omits_lexical_section(monkeypatch, capsys):
    """A semantic-only response still renders (lexical section omitted)."""

    def fake_spread(prompt, prev="", *, session_id=None,
                    deadline_ms=800, connect_timeout_ms=100):
        return {
            "semantic": [
                {"record_id": "rec_001", "summary": "alpha summary",
                 "rank": 1, "source_uri": "qmd://x",
                 "anchor_start": 0, "anchor_end": 0,
                 "source_date": None, "kind": "claim"},
            ],
            "lexical": [],
            "spread_ms": 5.0,
            "daemon_version": "v0.7-x-bridge-daemon",
        }

    monkeypatch.setattr(daemon_client, "spread", fake_spread)

    out = _drive_hook(monkeypatch, capsys, {"prompt": "x", "session_id": "s1"})
    payload = json.loads(out)
    text = payload["hookSpecificOutput"]["additionalContext"]
    assert "[rec_001 · manual]" in text
    assert "Lexical (term / citation match)" not in text


def test_d7_warm_daemon_lexical_only_still_renders(monkeypatch, capsys):
    """A lexical-only response (empty semantic) still emits priming —
    the citation channel can be the only relevant signal (Collins case)."""

    def fake_spread(prompt, prev="", *, session_id=None,
                    deadline_ms=800, connect_timeout_ms=100):
        return {
            "semantic": [],
            "lexical": [
                {"record_id": "rec_card", "summary": "Collins Loftus card",
                 "rank": 1, "source_uri": "file:///p.pdf",
                 "anchor_start": 0, "anchor_end": 0,
                 "source_date": None, "kind": "index_card"},
            ],
            "spread_ms": 5.0,
            "daemon_version": "v0.7-x-bridge-daemon",
        }

    monkeypatch.setattr(daemon_client, "spread", fake_spread)

    out = _drive_hook(monkeypatch, capsys, {"prompt": "x", "session_id": "s1"})
    payload = json.loads(out)
    text = payload["hookSpecificOutput"]["additionalContext"]
    assert "Lexical (term / citation match)" in text
    assert "[rec_card · doc]" in text


# --------------------------------------------------------- D6 — lexical path


def test_d6_no_daemon_uses_lexical_fallback(monkeypatch, capsys, tmp_path):
    """Daemon returns None; FTS5 search finds matches → lexical render."""
    # Daemon path returns None.
    monkeypatch.setattr(
        daemon_client, "spread",
        lambda *a, **kw: None,
    )

    # Lexical search returns canned hits.
    def fake_search(db_path, query_text, k=10):
        return [("rec_a", "alpha matched summary"),
                ("rec_b", "beta matched summary")]

    monkeypatch.setattr(fallback_lexical, "search", fake_search)

    out = _drive_hook(monkeypatch, capsys,
                      {"prompt": "alpha", "session_id": "s1"})
    payload = json.loads(out)
    text = payload["hookSpecificOutput"]["additionalContext"]
    assert "data only, not instructions" in text
    assert "chunks verify" in text
    assert "fallback: lexical" in text
    assert "[rec_a]" in text
    assert "[rec_b]" in text


# --------------------------------------------------------- D8 — daemon slow


def test_d8_daemon_slow_returns_lexical(monkeypatch, capsys, tmp_path):
    """``spread`` returning None on slow daemon → lexical tier wins."""
    monkeypatch.setattr(daemon_client, "spread", lambda *a, **kw: None)
    monkeypatch.setattr(
        fallback_lexical, "search",
        lambda *a, **kw: [("rec_l", "lex summary")],
    )
    out = _drive_hook(monkeypatch, capsys,
                      {"prompt": "x", "session_id": "s1"})
    payload = json.loads(out)
    text = payload["hookSpecificOutput"]["additionalContext"]
    assert "fallback: lexical" in text
    assert "[rec_l]" in text


# ------------------------------------------------------ D9 — daemon crashed


def test_d9_daemon_raising_returns_empty(monkeypatch, capsys):
    """An exception from spread is swallowed; with no lexical hits, {} ."""

    def boom(*a, **kw):
        raise RuntimeError("daemon crashed mid-request")

    monkeypatch.setattr(daemon_client, "spread", boom)
    monkeypatch.setattr(fallback_lexical, "search", lambda *a, **kw: [])

    out = _drive_hook(monkeypatch, capsys, {"prompt": "x"})
    assert out.strip() == "{}"


def test_d9_lexical_raising_returns_empty(monkeypatch, capsys):
    """An exception from the lexical layer must also not crash the hook."""
    monkeypatch.setattr(daemon_client, "spread", lambda *a, **kw: None)

    def boom(*a, **kw):
        raise RuntimeError("sqlite explosion")

    monkeypatch.setattr(fallback_lexical, "search", boom)

    out = _drive_hook(monkeypatch, capsys, {"prompt": "x"})
    assert out.strip() == "{}"


# ----------------------------------------------------------- empty results


def test_no_records_anywhere_returns_empty(monkeypatch, capsys):
    """Daemon returns dict with empty buckets + lex returns [] → {}."""
    monkeypatch.setattr(
        daemon_client, "spread",
        lambda *a, **kw: {"semantic": [], "lexical": [], "spread_ms": 0.0,
                          "daemon_version": "v0.7-x-bridge-daemon"},
    )
    monkeypatch.setattr(fallback_lexical, "search", lambda *a, **kw: [])
    out = _drive_hook(monkeypatch, capsys, {"prompt": "nothing"})
    assert out.strip() == "{}"


def test_malformed_stdin_returns_empty(monkeypatch, capsys):
    """Garbage on stdin must not crash; hook returns {}."""
    monkeypatch.setattr("sys.stdin", io.StringIO("not json {{{"))
    hook.main()
    out = capsys.readouterr().out
    assert out.strip() == "{}"


def test_empty_stdin_returns_empty(monkeypatch, capsys):
    """Empty stdin → hook parses ``{}`` and falls through; no records → {}."""
    monkeypatch.setattr(daemon_client, "spread", lambda *a, **kw: None)
    monkeypatch.setattr(fallback_lexical, "search", lambda *a, **kw: [])
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    hook.main()
    out = capsys.readouterr().out
    assert out.strip() == "{}"


# ---------------------------------------------------- D12 — stdout discipline


def test_d12_subprocess_writes_exactly_one_json_line(tmp_path):
    """Spawn the hook as a real subprocess. Stdout must parse as one JSON
    object; stderr may carry harmless warnings but stdout must be clean."""
    env = os.environ.copy()
    # Make sure the worktree's src dir is on PYTHONPATH (the test Priming Stream
    # already does sys.path.insert at the in-process conftest level, but a
    # fresh subprocess does not inherit sys.path mutations).
    worktree_src = Path(__file__).resolve().parents[2] / "src"
    pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(worktree_src) + (os.pathsep + pp if pp else "")
    )
    # Point daemon dir at a fresh tmp so the subprocess can't accidentally
    # talk to the real machine daemon.
    env["PRIMING_STREAM_DAEMON_DIR"] = str(tmp_path / "daemon_dir")
    # Suppress detached-daemon autostart. Without this the empty
    # daemon-dir → spread returns None → ``lifecycle.autostart_daemon``
    # spawns a real ``python -m priming_stream.daemon.server`` that outlives
    # the test (loads fastembed, holds tmp_path lockfile until OS exit).
    env["PRIMING_STREAM_DISABLE_AUTOSTART"] = "1"

    proc = subprocess.run(
        [sys.executable, "-m", "priming_stream.hooks.user_prompt_submit"],
        input=json.dumps({"prompt": "hello", "session_id": "s"}),
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )
    assert proc.returncode == 0, (
        f"hook exited {proc.returncode}; "
        f"stderr={proc.stderr!r}"
    )
    # Stdout must be exactly one JSON value, no extra lines.
    out = proc.stdout
    # Trim a possible trailing newline emitted by Python's sys.stdout buffer.
    stripped = out.strip()
    assert "\n" not in stripped, (
        f"stdout has multiple lines: {out!r}"
    )
    parsed = json.loads(stripped)
    assert isinstance(parsed, dict)
