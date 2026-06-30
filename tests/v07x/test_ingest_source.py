"""v0.7-x unify — ``prime ingest-source`` (source-ingest leg of the ingest
skill).

The load-bearing property: ingest-source writes chunks to the episodic store
but **does NOT materialize** — no ``corpus/imports/`` files, no ``_cursor.json``
advance. Materialize belongs to the cycle's ``sleep-prepare`` alone; splitting
it here removes the manual cursor-reset that ``coldstart`` forced before sleep
(W7 "critical glue").

Covers both adapter kinds (``cc`` validates the previously-unwired
``ClaudeCodeAdapter``), idempotent re-run, multiple ``--path``, and the
missing-graph guard. No LLM, no fastembed — pure plumbing.
"""
from __future__ import annotations

import json
from pathlib import Path

from priming_stream.cli.ingest_source import cmd_ingest_source
from priming_stream.core.db import connect
from priming_stream.core.episodic import EpisodicStore
from priming_stream.core.schema import apply_migrations


# -- fixtures -------------------------------------------------------------


def _init_storage(tmp_path: Path) -> Path:
    storage = tmp_path / "storage"
    storage.mkdir()
    conn = connect(storage / "graph.db")
    try:
        apply_migrations(conn)
    finally:
        conn.close()
    return storage


def _make_cc_session(tmp_path: Path, session_id: str = "sess-1") -> Path:
    """A minimal Claude Code transcript .jsonl (one session, a few turns)."""
    lines = [
        {
            "type": "user", "sessionId": session_id,
            "timestamp": "2026-06-04T10:00:00Z",
            "message": {"role": "user", "content": "idee despre arhitectura X"},
        },
        {
            "type": "assistant", "sessionId": session_id,
            "timestamp": "2026-06-04T10:00:30Z",
            "message": {"role": "assistant", "content": [
                {"type": "text", "text": "decizia: mergem cu varianta B"},
            ]},
        },
        {
            "type": "user", "sessionId": session_id,
            "timestamp": "2026-06-04T10:01:00Z",
            "message": {"role": "user", "content": "ok, confirm"},
        },
    ]
    d = tmp_path / "cc"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{session_id}.jsonl"
    f.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in lines) + "\n",
        encoding="utf-8",
    )
    return f


def _make_export_dir(tmp_path: Path, n_convs: int = 3) -> Path:
    convs = [{
        "uuid": f"u{i}", "name": f"conv-{i}",
        "created_at": "2026-01-01T00:00:00Z",
        "chat_messages": [
            {"sender": "human", "text": f"hello {i}",
             "created_at": "2026-01-01T00:00:00Z"},
            {"sender": "assistant", "text": f"hi {i}",
             "created_at": "2026-01-01T00:00:05Z"},
        ],
    } for i in range(n_convs)]
    d = tmp_path / "export"
    d.mkdir()
    (d / "conversations.json").write_text(
        json.dumps(convs, ensure_ascii=False), encoding="utf-8")
    return d


def _args(kind: str, *paths: str):
    return type("Args", (), {"kind": kind, "paths": list(paths)})()


def _chunk_count(storage: Path) -> int:
    return len(list(EpisodicStore(storage / "episodic").iter_chunks()))


# -- tests ----------------------------------------------------------------


def test_cc_ingest_writes_chunks_without_materializing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    storage = _init_storage(tmp_path)
    f = _make_cc_session(tmp_path)

    rc = cmd_ingest_source(_args("cc", str(f)))
    assert rc == 0

    # chunks landed in episodic
    assert _chunk_count(storage) >= 1
    chunks_jsonl = storage / "episodic" / "chunks.jsonl"
    assert chunks_jsonl.is_file() and chunks_jsonl.read_text(encoding="utf-8").strip()

    # KEY PROPERTY: no materialize — no imports/ files, no cursor advance.
    imports = storage / "corpus" / "imports"
    assert not imports.exists() or not list(imports.rglob("*.md"))
    assert not (storage / "corpus" / "_cursor.json").exists()


def test_export_ingest_writes_chunks_without_materializing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    storage = _init_storage(tmp_path)
    export_dir = _make_export_dir(tmp_path, n_convs=3)

    rc = cmd_ingest_source(_args("export", str(export_dir)))
    assert rc == 0

    assert _chunk_count(storage) >= 3
    imports = storage / "corpus" / "imports"
    assert not imports.exists() or not list(imports.rglob("*.md"))
    assert not (storage / "corpus" / "_cursor.json").exists()


def test_ingest_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    storage = _init_storage(tmp_path)
    f = _make_cc_session(tmp_path)

    assert cmd_ingest_source(_args("cc", str(f))) == 0
    first = _chunk_count(storage)
    assert cmd_ingest_source(_args("cc", str(f))) == 0
    assert _chunk_count(storage) == first  # no duplicates


def test_multiple_paths_accumulate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    storage = _init_storage(tmp_path)
    f1 = _make_cc_session(tmp_path, session_id="sess-A")
    f2 = _make_cc_session(tmp_path / "more", session_id="sess-B")

    rc = cmd_ingest_source(_args("cc", str(f1), str(f2)))
    assert rc == 0
    # two distinct sessions → at least two chunks
    assert _chunk_count(storage) >= 2


def test_missing_graph_db_errors(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    f = _make_cc_session(tmp_path)
    rc = cmd_ingest_source(_args("cc", str(f)))
    assert rc == 1
    assert "prime init" in capsys.readouterr().err


def test_missing_path_warns_not_fatal(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    storage = _init_storage(tmp_path)
    f = _make_cc_session(tmp_path)

    rc = cmd_ingest_source(_args("cc", str(tmp_path / "nope.jsonl"), str(f)))
    assert rc == 0  # missing path warns, the valid one still ingests
    assert "path missing" in capsys.readouterr().err
    assert _chunk_count(storage) >= 1
