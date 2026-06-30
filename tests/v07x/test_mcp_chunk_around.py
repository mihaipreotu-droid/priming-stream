"""v0.7-x W-F (F3): graph_chunk_around_anchor verification path.

Three fixtures:

- a record with anchor offsets pointing into a materialized chunk .md;
- a record with no anchor offsets (chunk-level record);
- a record id that doesn't exist.

We check the slice/window math, the full-file fallback, and the error
shape.
"""
from __future__ import annotations

from pathlib import Path

from priming_stream.core.db import connect
from priming_stream.core.graph_repo import GraphRepo
from priming_stream.core.models import Record, now_iso
from priming_stream.core.schema import apply_migrations
from priming_stream.graph_ops import graph_chunk_around_anchor


def _repo(tmp_path: Path) -> GraphRepo:
    conn = connect(tmp_path / "graph.db")
    apply_migrations(conn)
    return GraphRepo(conn)


def _write_imports_chunk(corpus_dir: Path, rel_path: str, text: str) -> Path:
    target = corpus_dir / "imports" / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8", newline="\n")
    return target


def test_chunk_around_returns_window_slice(tmp_path):
    """F3: anchor offsets present → window-bounded slice of file body."""
    corpus = tmp_path / "corpus"
    body = "A" * 100 + "TARGET_SLICE" + "B" * 100
    _write_imports_chunk(
        corpus, "claude_ai_export/sess1/chunk0.md", body,
    )

    repo = _repo(tmp_path)
    anchor_start = 100  # position of "T"
    anchor_end = 112    # one past the "E"
    repo.create_record(Record(
        id="rec_aaaaaaa1",
        source_uri="qmd://priming-stream-imports/claude_ai_export/sess1/chunk0.md",
        anchor_offset_start=anchor_start,
        anchor_offset_end=anchor_end,
        summary="contains target",
        created_at=now_iso(),
    ))

    out = graph_chunk_around_anchor(
        "rec_aaaaaaa1", window=10, repo=repo,
        storage_dir=tmp_path, corpus_dir=corpus,
    )
    assert out["record_id"] == "rec_aaaaaaa1"
    assert out["is_full_file"] is False
    assert "TARGET_SLICE" in out["text"]
    # Window of 10 on each side: text length should be 12 + 10 + 10 = 32.
    assert len(out["text"]) == 32


def test_chunk_around_null_anchors_returns_full_file(tmp_path):
    """F3: anchor_offset_start/end are None → return entire file body."""
    corpus = tmp_path / "corpus"
    body = "full file body content"
    _write_imports_chunk(
        corpus, "claude_ai_export/sess1/chunk_full.md", body,
    )

    repo = _repo(tmp_path)
    repo.create_record(Record(
        id="rec_bbbbbbb2",
        source_uri="qmd://priming-stream-imports/claude_ai_export/sess1/chunk_full.md",
        anchor_offset_start=None,
        anchor_offset_end=None,
        summary="whole-chunk record",
        created_at=now_iso(),
    ))

    out = graph_chunk_around_anchor(
        "rec_bbbbbbb2", window=200, repo=repo,
        storage_dir=tmp_path, corpus_dir=corpus,
    )
    assert out["is_full_file"] is True
    assert out["text"] == body


def test_chunk_around_missing_record_returns_error(tmp_path):
    repo = _repo(tmp_path)
    out = graph_chunk_around_anchor(
        "rec_deadbeef", window=200, repo=repo,
        storage_dir=tmp_path, corpus_dir=tmp_path / "corpus",
    )
    assert "error" in out
    assert "rec_deadbeef" in out["error"]


def test_chunk_around_window_clamps_at_zero(tmp_path):
    """Anchor near file start: window must not produce a negative index."""
    corpus = tmp_path / "corpus"
    body = "ABCDEFGHIJKLMN"
    _write_imports_chunk(
        corpus, "claude_ai_export/s/c.md", body,
    )
    repo = _repo(tmp_path)
    repo.create_record(Record(
        id="rec_cccccccc",
        source_uri="qmd://priming-stream-imports/claude_ai_export/s/c.md",
        anchor_offset_start=2,  # 'C'
        anchor_offset_end=4,    # past 'D'
        summary="near-start",
        created_at=now_iso(),
    ))
    out = graph_chunk_around_anchor(
        "rec_cccccccc", window=200, repo=repo,
        storage_dir=tmp_path, corpus_dir=corpus,
    )
    # Should return the whole body since the window dwarfs it.
    assert out["text"] == body
    assert out["is_full_file"] is False


def test_chunk_around_claude_code_session_self_anchored(tmp_path):
    """F-2: slash-command record with ``claude_code_session://`` source_uri.

    Self-anchored: no on-disk source. The handler returns the record's
    summary as ``text`` and flags ``self_anchored: True`` instead of an
    error.
    """
    repo = _repo(tmp_path)
    repo.create_record(Record(
        id="rec_dddddddd",
        source_uri="claude_code_session://sess-abc",
        anchor_offset_start=None,
        anchor_offset_end=None,
        summary="DECIS: use SQLite for the substrate",
        created_at=now_iso(),
    ))

    out = graph_chunk_around_anchor(
        "rec_dddddddd", window=200, repo=repo,
        storage_dir=tmp_path, corpus_dir=tmp_path / "corpus",
    )
    assert "error" not in out, out
    assert out["record_id"] == "rec_dddddddd"
    assert out["source_uri"] == "claude_code_session://sess-abc"
    assert out["text"] == "DECIS: use SQLite for the substrate"
    assert out["self_anchored"] is True
    assert out["is_full_file"] is False


def test_chunk_around_index_card_returns_body_not_source(tmp_path):
    """piece3: an index_card's source_uri points at the ORIGINAL document
    (often a binary PDF). The op must NOT try to read it as text (that
    raises an uncaught UnicodeDecodeError) — it returns the card body and
    the source link, like a self-anchored record."""
    # A real binary file at the source path, to prove we never read it.
    pdf = tmp_path / "papers" / "collins1975.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(b"%PDF-1.4\x00\xad\xde\xef binary not utf-8 \xff\xfe")

    repo = _repo(tmp_path)
    src = f"file:///{pdf.as_posix().lstrip('/')}"
    repo.create_record(Record(
        id="rec_card0001",
        source_uri=src,
        anchor_offset_start=0,
        anchor_offset_end=0,
        summary="## Summary\nSpreading-activation theory.\n\n## Key points\n- nodes + weighted links",
        created_at=now_iso(),
        kind="index_card",
        doc_key="path:" + pdf.resolve().as_posix(),
        source=src,
        content_hash="deadbeef",
    ))

    out = graph_chunk_around_anchor(
        "rec_card0001", window=200, repo=repo,
        storage_dir=tmp_path, corpus_dir=tmp_path / "corpus",
    )
    assert "error" not in out, out
    assert out["index_card"] is True
    assert out["text"].startswith("## Summary")
    assert "weighted links" in out["text"]
    assert out["source"] == src
    assert out["is_full_file"] is False


def test_chunk_around_file_uri_scheme(tmp_path):
    """In-place project doc via ``file://`` scheme is resolved literally."""
    project_doc = tmp_path / "project" / "doc.md"
    project_doc.parent.mkdir(parents=True, exist_ok=True)
    body = "abcdefghij" * 10  # 100 chars
    project_doc.write_text(body, encoding="utf-8", newline="\n")

    repo = _repo(tmp_path)
    repo.create_record(Record(
        id="rec_eeeeeeee",
        source_uri=f"file:///{project_doc.as_posix().lstrip('/')}",
        anchor_offset_start=10,
        anchor_offset_end=20,
        summary="in-place project doc",
        created_at=now_iso(),
    ))
    out = graph_chunk_around_anchor(
        "rec_eeeeeeee", window=5, repo=repo,
        storage_dir=tmp_path, corpus_dir=tmp_path / "corpus",
    )
    assert "error" not in out, out
    assert out["is_full_file"] is False
    # Window of 5 each side: 5 + 10 + 5 = 20.
    assert len(out["text"]) == 20
