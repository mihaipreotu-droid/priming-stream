"""Active-use telemetry (Phase-5 calibration input): writer + echo↔usage join.

Covers the usage.jsonl writer (per-tool field extraction, skip rule, session
from env, kill-switch, gated prune, dispatch integration) and the read-time
join in :mod:`priming_stream.core.usage_join` (attribution by session + timestamp,
fallback, orphans, signal classification).
"""
from __future__ import annotations

import json

import pytest

from priming_stream.core.db import connect
from priming_stream.core.graph_repo import GraphRepo
from priming_stream.core.models import Record
from priming_stream.core.schema import apply_migrations
from priming_stream.core.usage_join import (
    attach_usage_to_echoes,
    classify_usage,
    read_usage,
    role_for,
    surfaced_set,
)
from priming_stream.mcp_server import server as server_mod
from priming_stream.mcp_server import usage_log


def now_iso() -> str:
    return "2026-06-16T12:00:00Z"


@pytest.fixture
def usage_on(monkeypatch):
    """Enable the channel (conftest sets PRIMING_STREAM_USAGE_OFF globally) + a session."""
    monkeypatch.delenv("PRIMING_STREAM_USAGE_OFF", raising=False)
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "sess-A")


def _read_lines(path):
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


# -- writer: per-tool extraction ---------------------------------------------


def test_fetch_tool_logs_record_id(tmp_path, usage_on):
    graph_db = tmp_path / "graph.db"
    usage_log.log_usage(
        "graph_chunk_around_anchor", {"record_id": "rec_x"},
        {"text": "..."}, 4.2, graph_db,
    )
    lines = _read_lines(tmp_path / "episodic" / "usage.jsonl")
    assert len(lines) == 1
    line = lines[0]
    assert line["tool"] == "graph_chunk_around_anchor"
    assert line["record_id"] == "rec_x"
    assert line["query"] is None and line["result_ids"] is None
    assert line["session_id"] == "sess-A"
    assert line["elapsed_ms"] == 4.2


def test_search_tool_logs_query_mode_and_results(tmp_path, usage_on):
    graph_db = tmp_path / "graph.db"
    result = [{"record_id": "rec_a"}, {"record_id": "rec_b"}, {"no_id": 1}]
    usage_log.log_usage(
        "graph_search_lexical",
        {"query_text": "Collins Loftus", "mode": "or", "k": 10},
        result, 31.0, graph_db,
    )
    line = _read_lines(tmp_path / "episodic" / "usage.jsonl")[0]
    assert line["tool"] == "graph_search_lexical"
    assert line["query"] == "Collins Loftus"
    assert line["mode"] == "or"
    assert line["result_ids"] == ["rec_a", "rec_b"]
    assert line["record_id"] is None


def test_pull_markdown_tool_logs_query_no_results(tmp_path, usage_on):
    """graph_salient_context returns rendered markdown (a str) → no ids."""
    graph_db = tmp_path / "graph.db"
    usage_log.log_usage(
        "graph_salient_context", {"message": "what about the bridge"},
        "## Semantic\n- ...", 12.0, graph_db,
    )
    line = _read_lines(tmp_path / "episodic" / "usage.jsonl")[0]
    assert line["tool"] == "graph_salient_context"
    assert line["query"] == "what about the bridge"
    assert line["result_ids"] is None


def test_spread_tool_logs_text_and_results(tmp_path, usage_on):
    graph_db = tmp_path / "graph.db"
    usage_log.log_usage(
        "graph_spread", {"text": "acme attribution"},
        [{"record_id": "rec_z"}], 50.0, graph_db,
    )
    line = _read_lines(tmp_path / "episodic" / "usage.jsonl")[0]
    assert line["query"] == "acme attribution"
    assert line["result_ids"] == ["rec_z"]


def test_graph_stats_is_skipped(tmp_path, usage_on):
    graph_db = tmp_path / "graph.db"
    usage_log.log_usage("graph_stats", {}, {"records_count": 5}, 1.0, graph_db)
    assert not (tmp_path / "episodic" / "usage.jsonl").exists()


# -- writer: kill-switch + isolation -----------------------------------------


def test_kill_switch_suppresses_write(tmp_path, monkeypatch):
    monkeypatch.setenv("PRIMING_STREAM_USAGE_OFF", "1")
    graph_db = tmp_path / "graph.db"
    usage_log.log_usage(
        "graph_chunk_around_anchor", {"record_id": "rec_x"}, {}, 1.0, graph_db,
    )
    assert not (tmp_path / "episodic" / "usage.jsonl").exists()


def test_episodic_dir_derived_from_graph_db_parent(tmp_path, usage_on):
    graph_db = tmp_path / "nested" / "graph.db"
    usage_log.log_usage(
        "graph_records", {"record_id": "rec_q"}, {"id": "rec_q"}, 1.0, graph_db,
    )
    assert (tmp_path / "nested" / "episodic" / "usage.jsonl").exists()


def test_session_id_empty_when_env_absent(tmp_path, monkeypatch):
    monkeypatch.delenv("PRIMING_STREAM_USAGE_OFF", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    graph_db = tmp_path / "graph.db"
    usage_log.log_usage(
        "graph_records", {"record_id": "rec_q"}, None, 1.0, graph_db,
    )
    line = _read_lines(tmp_path / "episodic" / "usage.jsonl")[0]
    assert line["session_id"] == ""


# -- writer: gated prune ------------------------------------------------------


def test_prune_drops_lines_older_than_retention(tmp_path, usage_on):
    episodic = tmp_path / "episodic"
    episodic.mkdir()
    path = episodic / "usage.jsonl"
    # First (oldest) line predates the 30-day window → prune triggers.
    old = {"at": "2020-01-01T00:00:00Z", "tool": "graph_records",
           "record_id": "rec_old", "session_id": ""}
    recent = {"at": "2026-06-16T11:00:00Z", "tool": "graph_records",
              "record_id": "rec_recent", "session_id": ""}
    path.write_text(
        json.dumps(old) + "\n" + json.dumps(recent) + "\n", encoding="utf-8",
    )
    usage_log.log_usage(
        "graph_records", {"record_id": "rec_new"}, None, 1.0, tmp_path / "graph.db",
    )
    ids = [l.get("record_id") for l in _read_lines(path)]
    assert "rec_old" not in ids          # pruned
    assert "rec_recent" in ids           # kept
    assert "rec_new" in ids              # freshly appended


# -- writer: dispatch integration --------------------------------------------


def test_dispatch_tool_writes_usage_line(tmp_path, usage_on):
    """End-to-end: dispatching a real tool appends a usage line."""
    graph_db = tmp_path / "graph.db"
    conn = connect(graph_db)
    apply_migrations(conn)
    GraphRepo(conn).create_record(Record(
        id="rec_known", source_uri="owner://x", anchor_offset_start=None,
        anchor_offset_end=None, summary="known", created_at=now_iso(),
    ))
    conn.close()
    out = server_mod.dispatch_tool("graph_records", {"record_id": "rec_known"}, graph_db)
    assert out is not None and out["id"] == "rec_known"
    line = _read_lines(tmp_path / "episodic" / "usage.jsonl")[0]
    assert line["tool"] == "graph_records" and line["record_id"] == "rec_known"
    assert isinstance(line["elapsed_ms"], (int, float))


def test_dispatch_tool_usage_failure_never_breaks_call(tmp_path, usage_on, monkeypatch):
    """A telemetry exception must not surface to the tool caller.

    Even with a raising ``log_usage`` stub, the call-site guard keeps the tool
    result flowing — telemetry is best-effort, never load-bearing.
    """
    graph_db = tmp_path / "graph.db"
    conn = connect(graph_db)
    apply_migrations(conn)
    GraphRepo(conn).create_record(Record(
        id="rec_known", source_uri="owner://x", anchor_offset_start=None,
        anchor_offset_end=None, summary="known", created_at=now_iso(),
    ))
    conn.close()

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(server_mod, "log_usage", boom)
    out = server_mod.dispatch_tool(
        "graph_records", {"record_id": "rec_known"}, graph_db,
    )
    assert out is not None and out["id"] == "rec_known"


# -- join: taxonomy + reader --------------------------------------------------


def test_role_for_maps_tools():
    assert role_for("graph_chunk_around_anchor") == "fetch"
    assert role_for("graph_records") == "fetch"
    assert role_for("graph_search_records") == "search"
    assert role_for("graph_search_lexical") == "search"
    assert role_for("graph_spread") == "pull"
    assert role_for("graph_salient_context") == "pull"
    assert role_for("graph_stats") == "other"


def test_read_usage_tolerant(tmp_path):
    path = tmp_path / "usage.jsonl"
    path.write_text(
        json.dumps({"at": "2026-06-16T10:00:00Z", "tool": "graph_records"}) + "\n"
        + "{ broken\n" + "\n"
        + json.dumps({"at": "2026-06-16T11:00:00Z", "tool": "graph_spread"}) + "\n",
        encoding="utf-8",
    )
    out = read_usage(path)
    assert len(out) == 2
    assert read_usage(tmp_path / "absent.jsonl") == []


# -- join: attribution --------------------------------------------------------


def _echo(at, sid, semantic=None, lexical=None):
    return {"at": at, "session_id": sid, "semantic": semantic or [],
            "lexical": lexical or [], "prompt_head": "p"}


def _usage(at, sid, tool="graph_chunk_around_anchor", record_id=None, result_ids=None):
    return {"at": at, "session_id": sid, "tool": tool,
            "record_id": record_id, "result_ids": result_ids}


def test_attach_prefers_latest_same_session_before_usage():
    echoes = [
        _echo("2026-06-16T09:00:00Z", "sess-A"),
        _echo("2026-06-16T10:00:00Z", "sess-A"),
        _echo("2026-06-16T10:30:00Z", "sess-B"),
    ]
    usage = [_usage("2026-06-16T10:15:00Z", "sess-A", record_id="rec_x")]
    turns, orphans = attach_usage_to_echoes(echoes, usage)
    assert orphans == []
    # attaches to the 10:00 sess-A echo (latest sess-A at-or-before 10:15)
    attached = [t for t in turns if t["used"]]
    assert len(attached) == 1
    assert attached[0]["at"] == "2026-06-16T10:00:00Z"


def test_attach_falls_back_to_any_session_when_no_match():
    echoes = [_echo("2026-06-16T10:00:00Z", "sess-Y")]
    usage = [_usage("2026-06-16T10:30:00Z", "sess-X", record_id="rec_x")]
    turns, orphans = attach_usage_to_echoes(echoes, usage)
    assert orphans == []
    assert turns[0]["used"][0]["record_id"] == "rec_x"


def test_attach_orphans_usage_before_any_echo():
    echoes = [_echo("2026-06-16T10:00:00Z", "sess-A")]
    usage = [_usage("2026-06-16T09:00:00Z", "sess-A", record_id="rec_x")]
    turns, orphans = attach_usage_to_echoes(echoes, usage)
    assert len(orphans) == 1
    assert turns[0]["used"] == []


# -- join: signal classification ---------------------------------------------


def test_classify_verified_use():
    echo = _echo("t", "s", semantic=["rec_x"])
    u = _usage("t", "s", tool="graph_chunk_around_anchor", record_id="rec_x")
    assert classify_usage(u, echo) == "verified-use"


def test_classify_fetch_unprimed():
    echo = _echo("t", "s", semantic=["rec_y"])
    u = _usage("t", "s", tool="graph_records", record_id="rec_x")
    assert classify_usage(u, echo) == "fetch-unprimed"


def test_classify_recall_miss_when_search_surfaces_new_ids():
    echo = _echo("t", "s", semantic=["rec_a"])
    u = _usage("t", "s", tool="graph_search_records", result_ids=["rec_a", "rec_new"])
    assert classify_usage(u, echo) == "recall-miss"


def test_classify_search_when_all_results_were_primed():
    echo = _echo("t", "s", semantic=["rec_a"], lexical=["rec_b"])
    u = _usage("t", "s", tool="graph_search_records", result_ids=["rec_a", "rec_b"])
    assert classify_usage(u, echo) == "search"


def test_surfaced_set_unions_buckets():
    echo = _echo("t", "s", semantic=["rec_a"], lexical=["rec_b"])
    assert surfaced_set(echo) == {"rec_a", "rec_b"}
