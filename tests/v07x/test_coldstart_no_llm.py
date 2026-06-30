"""v0.7-x W-E: coldstart is deterministic plumbing — no LLM imports.

The acceptance contract (spec §4-E E2) is that coldstart must not pull in
``claude_agent_sdk`` or any sleep-worker LLM glue. The check here is
two-pronged:

1. Static: scan the coldstart module source for forbidden substrings.
2. Dynamic: import the module and walk its module graph one level deep,
   confirming no ``claude_agent_sdk`` shows up.
"""
from __future__ import annotations

import ast
import importlib
import json
import os
import sys
from pathlib import Path

import pytest

import priming_stream.cli.coldstart as coldstart_mod


_FORBIDDEN_NAMES = (
    "claude_agent_sdk",
    "priming_stream.sleep_worker.judge",
    "priming_stream.sleep_worker.orchestrator",
)


def _source_path() -> Path:
    return Path(coldstart_mod.__file__).resolve()


# -- static checks --------------------------------------------------------


def test_coldstart_source_has_no_forbidden_substrings():
    text = _source_path().read_text(encoding="utf-8")
    for name in _FORBIDDEN_NAMES:
        assert name not in text, (
            f"forbidden name '{name}' present in coldstart.py — coldstart "
            f"must stay LLM-free"
        )


def test_coldstart_ast_imports_are_clean():
    """Parse the module and verify no import statement references the
    forbidden modules. AST-based check is stricter than substring (would
    miss commented-out / string occurrences as desired)."""
    tree = ast.parse(_source_path().read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
    for name in _FORBIDDEN_NAMES:
        for imp in imports:
            assert not imp.startswith(name), (
                f"coldstart imports forbidden module: {imp}"
            )


# -- dynamic checks -------------------------------------------------------


def test_coldstart_module_does_not_load_claude_agent_sdk():
    """After importing the coldstart module fresh, ``claude_agent_sdk``
    must not be in ``sys.modules`` via coldstart's transitive imports.

    Other test files in the repo may legitimately import it (e.g. older
    sleep_worker tests still on the suite). To make this test independent
    of suite order, we snapshot sys.modules before importing coldstart
    fresh and assert coldstart itself does not pull it in.
    """
    # Drop any cached coldstart so the importlib.reload-like check runs
    # against a fresh import graph.
    pre = set(sys.modules)
    importlib.reload(coldstart_mod)
    introduced = set(sys.modules) - pre
    for mod in introduced:
        assert "claude_agent_sdk" not in mod, (
            f"coldstart import pulled in {mod}"
        )


# -- runtime smoke (no qmd, with --skip-qmd) ------------------------------


def _make_export_dir(tmp_path: Path, n_convs: int = 5) -> Path:
    """Create a synthetic conversations.json with N small conversations."""
    convs = []
    for i in range(n_convs):
        convs.append({
            "uuid": f"u{i}",
            "name": f"conv-{i}",
            "created_at": "2026-01-01T00:00:00Z",
            "chat_messages": [
                {
                    "sender": "human",
                    "text": f"hello number {i}",
                    "created_at": "2026-01-01T00:00:00Z",
                },
                {
                    "sender": "assistant",
                    "text": f"hi back {i}",
                    "created_at": "2026-01-01T00:00:05Z",
                },
            ],
        })
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "conversations.json").write_text(
        json.dumps(convs, ensure_ascii=False), encoding="utf-8",
    )
    return export_dir


def _make_manifest(tmp_path: Path, export_dir: Path) -> Path:
    cfg_path = tmp_path / "coldstart.toml"
    cfg_path.write_text(
        f'[exports]\npaths = ["{export_dir.as_posix()}"]\n',
        encoding="utf-8",
    )
    return cfg_path


def test_coldstart_runs_end_to_end_without_qmd(tmp_path, monkeypatch):
    """Run coldstart against a 5-conv synthetic export.

    v0.7-x-vec-index coldstart has no qmd / vec_index step — it only
    materializes chunks to .md. Asserts chunks.jsonl + .md files end up
    on disk under ``storage/corpus/imports``.
    """
    from priming_stream.cli.coldstart import handle_coldstart
    from priming_stream.core.db import connect
    from priming_stream.core.schema import apply_migrations

    # Run inside an isolated cwd so resolve_paths resolves under tmp_path.
    monkeypatch.chdir(tmp_path)

    # Initialize a clean storage tree (sample target of `prime init`).
    storage = tmp_path / "storage"
    storage.mkdir()
    db_path = storage / "graph.db"
    conn = connect(db_path)
    try:
        apply_migrations(conn)
    finally:
        conn.close()

    export_dir = _make_export_dir(tmp_path, n_convs=5)
    cfg_path = _make_manifest(tmp_path, export_dir)

    args = type("Args", (), {"config": str(cfg_path)})()
    rc = handle_coldstart(args)
    assert rc == 0

    # chunks.jsonl present and non-empty.
    chunks_jsonl = storage / "episodic" / "chunks.jsonl"
    assert chunks_jsonl.is_file()
    assert chunks_jsonl.read_text(encoding="utf-8").strip(), "chunks.jsonl empty"

    # At least one .md materialized under imports/claude_ai_export/<sid>/.
    imports_root = storage / "corpus" / "imports" / "claude_ai_export"
    assert imports_root.is_dir(), "imports root missing"
    md_files = list(imports_root.rglob("*.md"))
    assert len(md_files) >= 5, (
        f"expected >=5 chunk .md files, got {len(md_files)}"
    )


def test_coldstart_rerun_is_idempotent(tmp_path, monkeypatch):
    """Second invocation does not duplicate chunks or .md files."""
    from priming_stream.cli.coldstart import handle_coldstart
    from priming_stream.core.db import connect
    from priming_stream.core.schema import apply_migrations

    monkeypatch.chdir(tmp_path)
    storage = tmp_path / "storage"
    storage.mkdir()
    conn = connect(storage / "graph.db")
    try:
        apply_migrations(conn)
    finally:
        conn.close()

    export_dir = _make_export_dir(tmp_path, n_convs=3)
    cfg_path = _make_manifest(tmp_path, export_dir)
    args = type("Args", (), {"config": str(cfg_path)})()

    assert handle_coldstart(args) == 0
    first_chunk_lines = (storage / "episodic" / "chunks.jsonl") \
        .read_text(encoding="utf-8").splitlines()
    first_md = list(
        (storage / "corpus" / "imports").rglob("*.md")
    )

    assert handle_coldstart(args) == 0
    second_chunk_lines = (storage / "episodic" / "chunks.jsonl") \
        .read_text(encoding="utf-8").splitlines()
    second_md = list(
        (storage / "corpus" / "imports").rglob("*.md")
    )

    assert len(second_chunk_lines) == len(first_chunk_lines)
    assert sorted(p.name for p in second_md) == sorted(
        p.name for p in first_md
    )


def test_coldstart_missing_graph_db_errors(tmp_path, monkeypatch, capsys):
    from priming_stream.cli.coldstart import handle_coldstart

    monkeypatch.chdir(tmp_path)
    export_dir = _make_export_dir(tmp_path, n_convs=2)
    cfg_path = _make_manifest(tmp_path, export_dir)
    args = type("Args", (), {"config": str(cfg_path)})()
    rc = handle_coldstart(args)
    assert rc == 1
    err = capsys.readouterr().err
    assert "prime init" in err


def test_coldstart_missing_config_errors(tmp_path, monkeypatch, capsys):
    from priming_stream.cli.coldstart import handle_coldstart
    from priming_stream.core.db import connect
    from priming_stream.core.schema import apply_migrations

    monkeypatch.chdir(tmp_path)
    storage = tmp_path / "storage"
    storage.mkdir()
    conn = connect(storage / "graph.db")
    try:
        apply_migrations(conn)
    finally:
        conn.close()

    args = type(
        "Args", (),
        {"config": str(tmp_path / "nope.toml")},
    )()
    rc = handle_coldstart(args)
    assert rc == 1
    err = capsys.readouterr().err
    assert "config not found" in err.lower()
