"""v0.7-x unify F3 — produced/processed local documents → real cards.

The conversation branch surfaces a doc-candidate's ``LOCALPATH`` (a final/
substantial document the conversation produced or processed); ``writer.py``
collects those into ``_produced_docs.json``; ``doc_plan.py --originals-list``
feeds them (merged + deduped + doc-type-filtered) into the document branch so
they become REAL index cards in the same cycle.

These tests cover the two deterministic plumbing units:
1. ``writer._parse_doc_block`` parses the new ``LOCALPATH:`` field.
2. ``doc_plan.py --originals-list`` reads a JSON path list, filters to the
   document-type allowlist (a stray ``.py`` is dropped), and dedupes against
   ``--originals``.

Both skill scripts live under ``.claude/skills/`` (outside the package), so
they are loaded by path. Native ``.md`` originals avoid the markitdown step.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


WRITER = _load(REPO_ROOT / ".claude/skills/prime-ingest/writer.py", "ch_writer")
DOC_PLAN = _load(REPO_ROOT / ".claude/skills/prime-ingest/doc_plan.py", "ch_doc_plan")


# -- writer._parse_doc_block — LOCALPATH ---------------------------------


def test_parse_doc_block_reads_localpath():
    block = (
        "DOI:\nURL:\nAUTHORS:\nYEAR:\nDOCTITLE: Q4 Deck\n"
        "SOURCE:\nLOCALPATH: C:/work/q4_deck.pptx\n"
        "A pitch deck for the Q4 review."
    )
    f = WRITER._parse_doc_block(block)
    assert f is not None
    assert f["title"] == "Q4 Deck"
    assert f["local_path"] == "C:/work/q4_deck.pptx"
    assert f["stub"] == "A pitch deck for the Q4 review."


def test_parse_doc_block_no_localpath_is_none():
    block = "DOCTITLE: Some Paper\nA referenced paper, no local file."
    f = WRITER._parse_doc_block(block)
    assert f is not None
    assert f["local_path"] is None


# -- doc_plan --originals-list -------------------------------------------


def _init_storage(tmp_path: Path, monkeypatch):
    """Isolate storage via PRIMING_STREAM_STORAGE_DIR (how doc_plan's ``resolve_paths(cfg)``
    resolves — it takes no project_root). Returns the resolved paths."""
    from priming_stream.core.config import load_config
    from priming_stream.core.db import connect
    from priming_stream.core.paths import ensure_dirs, resolve_paths
    from priming_stream.core.schema import apply_migrations

    monkeypatch.setenv("PRIMING_STREAM_STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    paths = resolve_paths(cfg)
    ensure_dirs(paths)
    conn = connect(paths.graph_db)
    try:
        apply_migrations(conn)
    finally:
        conn.close()
    return paths


def test_doc_plan_originals_list_merges_filters_dedupes(tmp_path, monkeypatch, capsys):
    paths = _init_storage(tmp_path, monkeypatch)

    # Two scattered native-.md "documents" + one code file (must be dropped).
    a = tmp_path / "deckdir" / "report.md"
    b = tmp_path / "other" / "notes.md"
    code = tmp_path / "src" / "tool.py"
    for p, txt in ((a, "# Report\nfinal numbers"), (b, "# Notes\nclient brief"),
                   (code, "print('hi')")):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt, encoding="utf-8")

    produced = tmp_path / "produced.json"
    produced.write_text(
        json.dumps([str(a), str(b), str(code), str(a)]),  # dup a + code path
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys, "argv",
        ["doc_plan.py", "--originals-list", str(produced), "--no-generate"],
    )
    DOC_PLAN.main()

    out = capsys.readouterr().out
    # report.md + notes.md card; tool.py filtered; the duplicate collapses.
    assert "to_card=2" in out, out

    index_path = Path(paths.graph_db).parent / "corpus" / "_doc_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    sources = [d["source"] for d in index["docs"]]
    assert len(sources) == 2
    assert not any("tool.py" in s for s in sources), sources


def test_doc_plan_repeatable_originals(tmp_path, monkeypatch, capsys):
    """D.2: --originals is repeatable — several scattered folders/files in ONE
    cycle (a recursed dir + a single file), deduped, each enumerated once."""
    paths = _init_storage(tmp_path, monkeypatch)

    # A folder with two native-.md docs (recursed) + one scattered single file.
    folder = tmp_path / "papers"
    (folder / "sub").mkdir(parents=True)
    f1 = folder / "a.md"
    f2 = folder / "sub" / "b.md"
    scattered = tmp_path / "elsewhere" / "c.md"
    scattered.parent.mkdir(parents=True)
    for p, txt in ((f1, "# A\nfinal"), (f2, "# B\nbrief"), (scattered, "# C\nnote")):
        p.write_text(txt, encoding="utf-8")

    monkeypatch.setattr(
        sys, "argv",
        ["doc_plan.py", "--originals", str(folder),
         "--originals", str(scattered), "--no-generate"],
    )
    DOC_PLAN.main()

    out = capsys.readouterr().out
    assert "to_card=3" in out, out

    index_path = Path(paths.graph_db).parent / "corpus" / "_doc_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    sources = [d["source"] for d in index["docs"]]
    assert len(sources) == 3
    for name in ("a.md", "b.md", "c.md"):
        assert any(name in s for s in sources), (name, sources)


def test_doc_plan_repeatable_originals_dedupes_overlap(tmp_path, monkeypatch, capsys):
    """A file given both directly AND inside a passed folder is carded once."""
    _init_storage(tmp_path, monkeypatch)
    folder = tmp_path / "docs"
    folder.mkdir()
    f1 = folder / "x.md"
    f1.write_text("# X\nbody", encoding="utf-8")

    monkeypatch.setattr(
        sys, "argv",
        ["doc_plan.py", "--originals", str(folder),
         "--originals", str(f1), "--no-generate"],
    )
    DOC_PLAN.main()
    assert "to_card=1" in capsys.readouterr().out


def test_doc_plan_multi_root_conversions_map_per_root(tmp_path, monkeypatch, capsys):
    """D.2 conversions × multi-root — the one branch the other tests don't hit.
    Two original roots with DIFFERENT internal layouts share ONE --conversions
    root; each non-.md original must resolve its conversion against ITS OWN root
    (rel_root_for), not a single shared rel_root. With a single shared rel_root
    (the pre-fix bug) one original's relpath would mis-resolve and the conversion
    would be missed → --no-generate would skip it → to_card<2."""
    paths = _init_storage(tmp_path, monkeypatch)

    # Root A: p1.pdf at the root. Root B: p2.pdf one subdir deep. Different
    # layouts so a single shared rel_root cannot satisfy both.
    rootA = tmp_path / "origA"
    rootB = tmp_path / "origB"
    (rootA).mkdir()
    (rootB / "deep").mkdir(parents=True)
    p1 = rootA / "p1.pdf"
    p2 = rootB / "deep" / "p2.pdf"
    p1.write_bytes(b"%PDF-1 fake one")
    p2.write_bytes(b"%PDF-1 fake two")

    # Conversions root, parallel relpath per original's OWN root.
    conv = tmp_path / "conv"
    (conv / "deep").mkdir(parents=True)
    (conv / "p1.md").write_text("# P1\nfirst paper", encoding="utf-8")
    (conv / "deep" / "p2.md").write_text("# P2\nsecond paper", encoding="utf-8")

    monkeypatch.setattr(
        sys, "argv",
        ["doc_plan.py", "--originals", str(rootA), "--originals", str(rootB),
         "--conversions", str(conv), "--no-generate"],
    )
    DOC_PLAN.main()

    out = capsys.readouterr().out
    assert "to_card=2" in out, out
    assert "existing=2" in out, out          # both conversions found (not generated/skipped)
    assert "no_conversion_skipped=0" in out, out

    index_path = Path(paths.graph_db).parent / "corpus" / "_doc_index.json"
    sources = [d["source"] for d in
               json.loads(index_path.read_text(encoding="utf-8"))["docs"]]
    assert any("p1.pdf" in s for s in sources) and any("p2.pdf" in s for s in sources), sources


def test_doc_plan_requires_a_source(tmp_path, monkeypatch):
    _init_storage(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["doc_plan.py"])
    with pytest.raises(SystemExit):  # ap.error → SystemExit
        DOC_PLAN.main()


# -- writer: produced-doc carded even when NOT referenced (decouple fix) --


def test_writer_cards_unreferenced_produced_doc(tmp_path, monkeypatch):
    """A real-session run showed the worker's DOCREF not matching the doc's
    DOCTITLE, which orphan-dropped the produced deck. A doc-candidate whose
    LOCALPATH resolves to a real file must be carded REGARDLESS of whether a
    record DOCREFs it (the fragile DOCREF↔DOCTITLE match must not gate it)."""
    paths = _init_storage(tmp_path, monkeypatch)
    corpus = Path(paths.graph_db).parent / "corpus"
    for d in ("_sleep_assign", "_sleep_results", "records", "imports"):
        (corpus / d).mkdir(parents=True, exist_ok=True)
    deck = corpus / "deck.pptx"
    deck.write_text("x", encoding="utf-8")
    chunk = corpus / "imports" / "c1.md"
    chunk.write_text("---\nx: 1\n---\nbody text", encoding="utf-8")
    (corpus / "_sleep_assign" / "p.json").write_text(json.dumps({
        "rec_ids": ["rec_d1", "rec_d2"], "created_at": "2026-06-05T12:00:00Z",
        "chunks": [{"chunk_id": "c1", "source_uri": "owner://x",
                    "path": str(chunk)}]}), encoding="utf-8")
    # the record carries NO DOCREF → the doc is NOT referenced
    res = (
        "CONV: p\nNOTABLE: yes\nNOTE: t\n"
        "===REC===\nCHUNK: c1\nANCHOR: 0 9\nDECIS: built the deck\n"
        "===DOC===\nDOCTITLE: Q4 Deck\nSOURCE:\n"
        f"LOCALPATH: {deck}\nA pitch deck.\n"
    )
    (corpus / "_sleep_results" / "p.txt").write_text(res, encoding="utf-8")

    WRITER.main()

    produced = json.loads(
        (corpus / "_produced_docs.json").read_text(encoding="utf-8"))
    assert any("deck.pptx" in p for p in produced), produced


# -- resolution tuning: tool-event paths + subdir walk -------------------


def test_resolve_via_tool_event_path(tmp_path):
    """An INPUT doc read from outside cwd (e.g. Downloads): only its full path
    is in the session's tool-event doc_paths; resolution matches by basename."""
    rep = tmp_path / "Downloads" / "client_report.pdf"
    rep.parent.mkdir(parents=True)
    rep.write_text("x", encoding="utf-8")
    cwd = tmp_path / "project"
    cwd.mkdir()
    got = WRITER._resolve_doc_path("client_report.pdf", str(cwd), [str(rep)])
    assert got is not None and Path(got) == rep.resolve()
    # without the tool path, cwd-only resolution fails (not in cwd)
    assert WRITER._resolve_doc_path("client_report.pdf", str(cwd), []) is None


def test_resolve_via_subdir_walk(tmp_path):
    cwd = tmp_path / "proj"
    (cwd / "exports").mkdir(parents=True)
    deck = cwd / "exports" / "deck.pptx"
    deck.write_text("x", encoding="utf-8")
    # in a subdir, not in cwd root, not in doc_paths → bounded walk finds it
    got = WRITER._resolve_doc_path("deck.pptx", str(cwd), None)
    assert got is not None and Path(got) == deck.resolve()
    # heavy dirs are skipped by the walk
    (cwd / "node_modules").mkdir()
    (cwd / "node_modules" / "secret.pptx").write_text("x", encoding="utf-8")
    assert WRITER._resolve_doc_path("secret.pptx", str(cwd), None) is None


def test_cc_adapter_captures_doc_paths(tmp_path):
    from priming_stream.ingest.claude_code import ClaudeCodeAdapter

    sess = tmp_path / "s.jsonl"
    lines = [
        {"type": "user", "sessionId": "s1", "cwd": "C:/w",
         "timestamp": "2026-06-04T10:00:00Z",
         "message": {"role": "user", "content": "go"}},
        {"type": "assistant", "sessionId": "s1", "cwd": "C:/w",
         "timestamp": "2026-06-04T10:00:30Z",
         "message": {"role": "assistant", "content": [
             {"type": "text", "text": "working"},
             {"type": "tool_use", "name": "Read",
              "input": {"file_path": "C:/clients/acme/report.pdf"}},
             {"type": "tool_use", "name": "Write",
              "input": {"file_path": "C:/w/deck.pptx"}},
             {"type": "tool_use", "name": "Edit",
              "input": {"file_path": "C:/w/cli/main.py"}},  # code → excluded
         ]}},
    ]
    sess.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    dp = list(ClaudeCodeAdapter(sess).iter_chunks())[0].doc_paths
    assert "C:/clients/acme/report.pdf" in dp
    assert "C:/w/deck.pptx" in dp
    assert not any("main.py" in p for p in dp), dp  # code excluded


def test_doc_paths_roundtrip_materialize(tmp_path):
    from priming_stream.core.models import Chunk, Turn
    from priming_stream.ingest.materialize import materialize_chunk

    ch = Chunk(
        chunk_id="c1", source_client="claude_code", session_id="s1",
        started_at="2026-06-04T10:00:00Z", ended_at="2026-06-04T10:01:00Z",
        turns=[Turn(index=0, role="user", text="hi",
                    timestamp="2026-06-04T10:00:00Z")],
        cwd="C:/w", doc_paths=["C:/clients/report.pdf", "C:/w/deck.pptx"],
    )
    md = materialize_chunk(ch, tmp_path / "imports")
    assert WRITER._chunk_doc_paths(str(md)) == [
        "C:/clients/report.pdf", "C:/w/deck.pptx"]


# -- recall fix: basename + session cwd → full path ----------------------


def test_resolve_doc_path_basename_against_cwd(tmp_path):
    """The load-bearing recall fix: a produced-doc's full path is NOT in the
    conversation text (only the basename is; the path lives in a stripped
    tool_use / is skill-produced). Resolution joins the session cwd."""
    deck = tmp_path / "Q4 Strategy.pptx"
    deck.write_text("x", encoding="utf-8")

    # basename only + cwd → resolves to the real file
    got = WRITER._resolve_doc_path("Q4 Strategy.pptx", str(tmp_path))
    assert got is not None and Path(got) == deck.resolve()

    # an existing absolute path → used as-is (no cwd needed)
    assert Path(WRITER._resolve_doc_path(str(deck), None)) == deck.resolve()

    # nonexistent basename → None (filesystem is ground truth; honest drop)
    assert WRITER._resolve_doc_path("nope.pptx", str(tmp_path)) is None
    # no cwd + bare basename → None
    assert WRITER._resolve_doc_path("Q4 Strategy.pptx", None) is None
    # empty → None
    assert WRITER._resolve_doc_path(None, str(tmp_path)) is None


def test_cc_adapter_captures_cwd(tmp_path):
    """ClaudeCodeAdapter must carry each session's ``cwd`` onto its chunks so
    the writer can resolve produced-doc basenames."""
    from priming_stream.ingest.claude_code import ClaudeCodeAdapter

    sess = tmp_path / "s.jsonl"
    sess.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in [
        {"type": "user", "sessionId": "s1", "cwd": "C:/work/proj",
         "timestamp": "2026-06-04T10:00:00Z",
         "message": {"role": "user", "content": "make the deck"}},
        {"type": "assistant", "sessionId": "s1", "cwd": "C:/work/proj",
         "timestamp": "2026-06-04T10:00:30Z",
         "message": {"role": "assistant", "content": [
             {"type": "text", "text": "done, saved Deck.pptx"}]}},
    ]) + "\n", encoding="utf-8")

    chunks = list(ClaudeCodeAdapter(sess).iter_chunks())
    assert chunks and all(c.cwd == "C:/work/proj" for c in chunks)


def test_chunk_cwd_roundtrips_and_materializes(tmp_path):
    """cwd survives dict round-trip and lands in the materialized .md
    frontmatter (where the writer reads it back via _chunk_cwd)."""
    from priming_stream.core.models import (
        Chunk, Turn, chunk_from_dict, chunk_to_dict,
    )
    from priming_stream.ingest.materialize import materialize_chunk

    ch = Chunk(
        chunk_id="c1", source_client="claude_code", session_id="s1",
        started_at="2026-06-04T10:00:00Z", ended_at="2026-06-04T10:01:00Z",
        turns=[Turn(index=0, role="user", text="hi",
                    timestamp="2026-06-04T10:00:00Z")],
        cwd="C:/work/proj",
    )
    assert chunk_from_dict(chunk_to_dict(ch)).cwd == "C:/work/proj"

    md = materialize_chunk(ch, tmp_path / "imports")
    assert "cwd: C:/work/proj" in md.read_text(encoding="utf-8")
    assert WRITER._chunk_cwd(str(md)) == "C:/work/proj"
