"""v0.7-x W-B: materialize Chunk -> .md, cursor-driven drain, idempotent."""
from __future__ import annotations

import json
from pathlib import Path

from priming_stream.core.episodic import EpisodicStore
from priming_stream.core.models import Chunk, Turn
from priming_stream.ingest.materialize import (
    materialize_chunk,
    materialize_pending,
    safe_filename,
)


def _chunk(chunk_id: str = "export_abc_p0", n_turns: int = 2) -> Chunk:
    turns = []
    for i in range(n_turns):
        role = "user" if i % 2 == 0 else "assistant"
        turns.append(Turn(
            index=i,
            role=role,
            text=f"turn-{i} payload",
            timestamp=f"2026-05-25T10:00:{i:02d}Z",
        ))
    return Chunk(
        chunk_id=chunk_id,
        source_client="claude_ai_export",
        session_id="88521352-0e0a-484b-83bf-a4409ae92760",
        started_at="2026-05-25T10:00:00Z",
        ended_at="2026-05-25T10:01:00Z",
        turns=turns,
    )


# -- safe_filename --------------------------------------------------------


def test_safe_filename_replaces_illegal():
    assert safe_filename("foo:bar?baz") == "foo_bar_baz"


def test_safe_filename_replaces_all_illegal():
    # < > : " / \ | ? *
    assert safe_filename('<>:"/\\|?*') == "_________"


def test_safe_filename_trims_trailing_dot():
    assert safe_filename("foo.") == "foo"


def test_safe_filename_trims_trailing_space():
    assert safe_filename("foo ") == "foo"


def test_safe_filename_keeps_safe_chars():
    assert safe_filename("export_88521352-0e0a-484b_p0") == \
        "export_88521352-0e0a-484b_p0"


def test_safe_filename_empty_becomes_underscore():
    assert safe_filename("") == "_"


# -- materialize_chunk ----------------------------------------------------


def test_materialize_chunk_path_layout(tmp_path):
    chunk = _chunk()
    out = materialize_chunk(chunk, tmp_path / "imports")
    assert out == (
        tmp_path / "imports"
        / "claude_ai_export"
        / "88521352-0e0a-484b-83bf-a4409ae92760"
        / "export_abc_p0.md"
    )
    assert out.exists()


def test_materialize_chunk_frontmatter(tmp_path):
    chunk = _chunk(n_turns=3)
    out = materialize_chunk(chunk, tmp_path / "imports")
    content = out.read_text(encoding="utf-8")

    # Frontmatter block delimited by '---' on both sides.
    assert content.startswith("---\n")
    end = content.find("\n---\n", 4)
    assert end > 0
    fm = content[4:end]
    assert "chunk_id: export_abc_p0" in fm
    assert "session_id: 88521352-0e0a-484b-83bf-a4409ae92760" in fm
    assert "source_client: claude_ai_export" in fm
    assert "started_at: 2026-05-25T10:00:00Z" in fm
    assert "ended_at: 2026-05-25T10:01:00Z" in fm
    assert "turn_count: 3" in fm
    # 3 turns of 'turn-N payload' = 'turn-0 payload' (14) * 3 = 42
    assert "total_chars: 42" in fm


def test_materialize_chunk_body_format(tmp_path):
    chunk = _chunk(n_turns=2)
    out = materialize_chunk(chunk, tmp_path / "imports")
    content = out.read_text(encoding="utf-8")
    assert "## User — 2026-05-25T10:00:00Z" in content
    assert "## Assistant — 2026-05-25T10:00:01Z" in content
    assert "turn-0 payload" in content
    assert "turn-1 payload" in content


def test_materialize_chunk_idempotent(tmp_path):
    chunk = _chunk()
    out1 = materialize_chunk(chunk, tmp_path / "imports")
    mtime1 = out1.stat().st_mtime_ns
    # Re-run: content identical, no rewrite (mtime unchanged within same FS).
    out2 = materialize_chunk(chunk, tmp_path / "imports")
    assert out2 == out1
    assert out2.stat().st_mtime_ns == mtime1


# -- materialize_pending --------------------------------------------------


def test_materialize_pending_drains_all(tmp_path):
    episodic = EpisodicStore(tmp_path / "episodic")
    for i in range(3):
        episodic.write_chunk(_chunk(chunk_id=f"export_x_p{i}"))

    imports = tmp_path / "qmd-corpus" / "imports"
    cursor = tmp_path / "qmd-corpus" / "_cursor.json"
    written = materialize_pending(episodic, imports, cursor)
    assert len(written) == 3
    assert all(p.exists() for p in written)
    # Cursor advanced to last chunk_id seen.
    state = json.loads(cursor.read_text(encoding="utf-8"))
    assert state["last_chunk_id"] == "export_x_p2"
    assert "updated_at" in state


def test_materialize_pending_rerun_is_noop(tmp_path):
    episodic = EpisodicStore(tmp_path / "episodic")
    for i in range(3):
        episodic.write_chunk(_chunk(chunk_id=f"export_x_p{i}"))

    imports = tmp_path / "qmd-corpus" / "imports"
    cursor = tmp_path / "qmd-corpus" / "_cursor.json"
    first = materialize_pending(episodic, imports, cursor)
    assert len(first) == 3
    cursor_text_after_first = cursor.read_text(encoding="utf-8")

    second = materialize_pending(episodic, imports, cursor)
    assert second == []
    assert cursor.read_text(encoding="utf-8") == cursor_text_after_first


def test_materialize_pending_picks_up_new_chunks(tmp_path):
    episodic = EpisodicStore(tmp_path / "episodic")
    for i in range(2):
        episodic.write_chunk(_chunk(chunk_id=f"export_x_p{i}"))

    imports = tmp_path / "qmd-corpus" / "imports"
    cursor = tmp_path / "qmd-corpus" / "_cursor.json"
    first = materialize_pending(episodic, imports, cursor)
    assert len(first) == 2

    # Add two more chunks, drain again.
    for i in range(2, 4):
        episodic.write_chunk(_chunk(chunk_id=f"export_x_p{i}"))

    second = materialize_pending(episodic, imports, cursor)
    assert len(second) == 2
    paths = sorted(p.name for p in second)
    assert paths == ["export_x_p2.md", "export_x_p3.md"]

    state = json.loads(cursor.read_text(encoding="utf-8"))
    assert state["last_chunk_id"] == "export_x_p3"


def test_materialize_pending_respects_limit(tmp_path):
    episodic = EpisodicStore(tmp_path / "episodic")
    for i in range(5):
        episodic.write_chunk(_chunk(chunk_id=f"export_x_p{i}"))

    imports = tmp_path / "qmd-corpus" / "imports"
    cursor = tmp_path / "qmd-corpus" / "_cursor.json"
    written = materialize_pending(episodic, imports, cursor, limit=2)
    assert len(written) == 2
    state = json.loads(cursor.read_text(encoding="utf-8"))
    assert state["last_chunk_id"] == "export_x_p1"

    # Next call drains the rest.
    rest = materialize_pending(episodic, imports, cursor)
    assert len(rest) == 3


def test_materialize_pending_missing_cursor_drains_everything(tmp_path):
    episodic = EpisodicStore(tmp_path / "episodic")
    for i in range(2):
        episodic.write_chunk(_chunk(chunk_id=f"export_x_p{i}"))
    imports = tmp_path / "qmd-corpus" / "imports"
    cursor = tmp_path / "qmd-corpus" / "_cursor.json"
    assert not cursor.exists()
    written = materialize_pending(episodic, imports, cursor)
    assert len(written) == 2
    assert cursor.exists()


def test_materialize_pending_stale_cursor_yields_empty(tmp_path):
    """A cursor pointing at a chunk id we never see leaves the cursor
    alone and returns an empty list (defensive: chunks.jsonl may have been
    rewritten / truncated during testing)."""
    episodic = EpisodicStore(tmp_path / "episodic")
    for i in range(2):
        episodic.write_chunk(_chunk(chunk_id=f"export_x_p{i}"))
    imports = tmp_path / "qmd-corpus" / "imports"
    cursor = tmp_path / "qmd-corpus" / "_cursor.json"
    cursor.parent.mkdir(parents=True, exist_ok=True)
    cursor.write_text(
        json.dumps({"last_chunk_id": "never-seen", "updated_at": "2026-01-01T00:00:00Z"}),
        encoding="utf-8",
    )
    written = materialize_pending(episodic, imports, cursor)
    assert written == []
    # Cursor unchanged.
    state = json.loads(cursor.read_text(encoding="utf-8"))
    assert state["last_chunk_id"] == "never-seen"
