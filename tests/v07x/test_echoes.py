"""E.1 memory echoes — hook capture (echoes.jsonl) + 30-day prune +
``prime echoes`` CLI.

Capture fidelity is the design center: the echo records what THIS hook
invocation actually injected (daemon / fallback / empty), ids-only.
Conventions mirror test_hook_userpromptsubmit.py (stdin drive + monkeypatch
on the hook module's imported names).
"""
from __future__ import annotations

import argparse
import io
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from priming_stream.core.db import connect
from priming_stream.core.graph_repo import GraphRepo
from priming_stream.core.models import Record
from priming_stream.core.schema import apply_migrations
from priming_stream.daemon import client as daemon_client
from priming_stream.daemon import fallback_lexical
from priming_stream.hooks import user_prompt_submit as hook
from priming_stream.cli import echoes as echoes_cli

_AT_FMT = "%Y-%m-%dT%H:%M:%SZ"


def _at(days_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime(_AT_FMT)


def _echo_line(days_ago: float, **over) -> str:
    base = {
        "at": _at(days_ago), "session_id": "sess-test",
        "prompt_head": "p", "semantic": [], "lexical": [],
        "source": "daemon", "spread_ms": 10.0,
    }
    base.update(over)
    return json.dumps(base)


def _wire_paths(monkeypatch, tmp_path):
    """Point the hook's config/paths at tmp and enable the echo channel."""
    monkeypatch.delenv("PRIMING_STREAM_ECHOES_OFF", raising=False)
    paths = SimpleNamespace(
        episodic_dir=tmp_path / "episodic",
        graph_db=tmp_path / "graph.db",
    )
    monkeypatch.setattr(hook, "load_config", lambda *a, **kw: object())
    monkeypatch.setattr(hook, "resolve_paths", lambda *a, **kw: paths)
    return paths


def _drive(monkeypatch, capsys, event: dict) -> str:
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(event)))
    hook.main()
    return capsys.readouterr().out


def _read_lines(path):
    return [
        json.loads(ln)
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]


# -- capture ----------------------------------------------------------------


def test_echo_written_on_daemon_tier(monkeypatch, capsys, tmp_path):
    paths = _wire_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(daemon_client, "spread", lambda *a, **kw: {
        "semantic": [
            {"record_id": "rec_aaa", "summary": "s1"},
            {"record_id": "rec_bbb", "summary": "s2"},
        ],
        "lexical": [{"record_id": "rec_ccc", "summary": "s3"}],
        "spread_ms": 123.4,
    })
    out = _drive(monkeypatch, capsys, {"prompt": "hello world", "session_id": "s9"})
    assert "Salient context" in out

    lines = _read_lines(paths.episodic_dir / "echoes.jsonl")
    assert len(lines) == 1
    e = lines[0]
    assert e["source"] == "daemon"
    assert e["semantic"] == ["rec_aaa", "rec_bbb"]
    assert e["lexical"] == ["rec_ccc"]
    assert e["spread_ms"] == 123.4
    assert e["session_id"] == "s9"
    assert e["prompt_head"] == "hello world"


def test_echo_written_on_fallback_tier(monkeypatch, capsys, tmp_path):
    paths = _wire_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(daemon_client, "spread", lambda *a, **kw: None)
    monkeypatch.setattr(
        fallback_lexical, "search",
        lambda *a, **kw: [("rec_lex", "summary text")],
    )
    _drive(monkeypatch, capsys, {"prompt": "x", "session_id": "s1"})

    e = _read_lines(paths.episodic_dir / "echoes.jsonl")[0]
    assert e["source"] == "fallback"
    assert e["semantic"] == []
    assert e["lexical"] == ["rec_lex"]


def test_echo_written_on_empty_tier(monkeypatch, capsys, tmp_path):
    paths = _wire_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(daemon_client, "spread", lambda *a, **kw: None)
    monkeypatch.setattr(fallback_lexical, "search", lambda *a, **kw: [])
    out = _drive(monkeypatch, capsys, {"prompt": "x"})
    assert out == "{}"

    e = _read_lines(paths.episodic_dir / "echoes.jsonl")[0]
    assert e["source"] == "empty"
    assert e["semantic"] == [] and e["lexical"] == []


def test_echo_failure_never_breaks_the_turn(monkeypatch, capsys, tmp_path):
    """A broken echo path must not change hook output (best-effort)."""
    monkeypatch.delenv("PRIMING_STREAM_ECHOES_OFF", raising=False)

    def boom(*a, **kw):
        raise RuntimeError("no paths")

    monkeypatch.setattr(hook, "resolve_paths", boom)
    monkeypatch.setattr(daemon_client, "spread", lambda *a, **kw: {
        "semantic": [{"record_id": "rec_aaa", "summary": "s"}],
        "lexical": [],
        "spread_ms": 1.0,
    })
    out = _drive(monkeypatch, capsys, {"prompt": "x"})
    assert "Salient context" in out  # priming unaffected


def test_echo_off_switch(monkeypatch, capsys, tmp_path):
    paths = _wire_paths(monkeypatch, tmp_path)
    monkeypatch.setenv("PRIMING_STREAM_ECHOES_OFF", "1")
    monkeypatch.setattr(daemon_client, "spread", lambda *a, **kw: None)
    monkeypatch.setattr(fallback_lexical, "search", lambda *a, **kw: [])
    _drive(monkeypatch, capsys, {"prompt": "x"})
    assert not (paths.episodic_dir / "echoes.jsonl").exists()


# -- prune (30-day retention) -------------------------------------------------


def test_prune_drops_old_lines_on_append(monkeypatch, capsys, tmp_path):
    paths = _wire_paths(monkeypatch, tmp_path)
    echoes = paths.episodic_dir / "echoes.jsonl"
    echoes.parent.mkdir(parents=True)
    echoes.write_text(
        _echo_line(40) + "\n" + _echo_line(35) + "\n" + _echo_line(5) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(daemon_client, "spread", lambda *a, **kw: None)
    monkeypatch.setattr(fallback_lexical, "search", lambda *a, **kw: [])
    _drive(monkeypatch, capsys, {"prompt": "new"})

    lines = _read_lines(echoes)
    # 40d + 35d pruned; 5d kept; new appended.
    assert len(lines) == 2
    assert lines[-1]["prompt_head"] == "new"


def test_prune_gate_skips_rewrite_when_first_line_recent(
    monkeypatch, capsys, tmp_path,
):
    """First line in-window → no rewrite (a corrupt later line survives,
    proving the O(file) pass did not run)."""
    paths = _wire_paths(monkeypatch, tmp_path)
    echoes = paths.episodic_dir / "echoes.jsonl"
    echoes.parent.mkdir(parents=True)
    echoes.write_text(
        _echo_line(3) + "\n" + "{corrupt json\n", encoding="utf-8",
    )
    monkeypatch.setattr(daemon_client, "spread", lambda *a, **kw: None)
    monkeypatch.setattr(fallback_lexical, "search", lambda *a, **kw: [])
    _drive(monkeypatch, capsys, {"prompt": "new"})

    raw = echoes.read_text(encoding="utf-8")
    assert "{corrupt json" in raw  # untouched: gate skipped the rewrite
    assert raw.strip().endswith('}')  # new line appended after it


# -- CLI ----------------------------------------------------------------------


def _cli_project(monkeypatch, tmp_path):
    """Wire echoes_cli paths at tmp + a migrated graph.db with one record."""
    paths = SimpleNamespace(
        episodic_dir=tmp_path / "episodic",
        graph_db=tmp_path / "graph.db",
    )
    monkeypatch.setattr(echoes_cli, "load_config", lambda *a, **kw: object())
    monkeypatch.setattr(echoes_cli, "resolve_paths", lambda *a, **kw: paths)
    paths.episodic_dir.mkdir(parents=True)

    conn = connect(paths.graph_db)
    apply_migrations(conn)
    GraphRepo(conn).create_record(Record(
        id="rec_known",
        source_uri="owner://",
        anchor_offset_start=0,
        anchor_offset_end=0,
        summary="known summary for echoes",
        created_at="2026-06-01T00:00:00Z",
        source_date="2026-06-01T00:00:00Z",
    ))
    conn.close()
    return paths


def test_cli_renders_resolved_summaries(monkeypatch, capsys, tmp_path):
    paths = _cli_project(monkeypatch, tmp_path)
    (paths.episodic_dir / "echoes.jsonl").write_text(
        _echo_line(1, semantic=["rec_known"], lexical=["rec_gone"],
                   prompt_head="ce am primat?")
        + "\n" + "{corrupt\n",
        encoding="utf-8",
    )
    rc = echoes_cli._cmd_echoes(argparse.Namespace(
        last=5, session=None, ids_only=False,
    ))
    out = capsys.readouterr().out
    assert rc == 0
    assert "known summary for echoes" in out
    assert "rec_gone (deleted)" in out
    assert "ce am primat?" in out


def test_cli_session_filter_and_last(monkeypatch, capsys, tmp_path):
    paths = _cli_project(monkeypatch, tmp_path)
    (paths.episodic_dir / "echoes.jsonl").write_text(
        _echo_line(3, session_id="aaa-1", prompt_head="one") + "\n"
        + _echo_line(2, session_id="bbb-2", prompt_head="two") + "\n"
        + _echo_line(1, session_id="bbb-2", prompt_head="three") + "\n",
        encoding="utf-8",
    )
    rc = echoes_cli._cmd_echoes(argparse.Namespace(
        last=1, session="bbb", ids_only=True,
    ))
    out = capsys.readouterr().out
    assert rc == 0
    assert "three" in out and "two" not in out and "one" not in out


def test_cli_no_echoes_message(monkeypatch, capsys, tmp_path):
    paths = SimpleNamespace(
        episodic_dir=tmp_path / "episodic",
        graph_db=tmp_path / "graph.db",
    )
    monkeypatch.setattr(echoes_cli, "load_config", lambda *a, **kw: object())
    monkeypatch.setattr(echoes_cli, "resolve_paths", lambda *a, **kw: paths)
    rc = echoes_cli._cmd_echoes(argparse.Namespace(
        last=5, session=None, ids_only=True,
    ))
    assert rc == 0
    assert "no echoes" in capsys.readouterr().out
