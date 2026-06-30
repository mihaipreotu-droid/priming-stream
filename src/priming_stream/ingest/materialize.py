"""Materialize ``Chunk`` objects to ``.md`` files under the corpus.

Layout (§4.2 / §16.10): one file per chunk, path

    <imports_root>/<source_client>/<session_uuid>/<chunk_id>.md

Frontmatter is a small fixed YAML block; body is each turn rendered as

    ## <Role> — <timestamp>

    <text>

separated by blank lines. The renderer doesn't re-flow whitespace inside
turn text.

Idempotency: ``materialize_pending`` consults a JSON cursor at
``storage/corpus/_cursor.json`` (renamed from ``qmd-corpus`` in
v0.7-x-vec-index). Re-running on an already-drained ``chunks.jsonl``
returns an empty list and leaves the cursor where it was.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from priming_stream.core.episodic import EpisodicStore
from priming_stream.core.models import Chunk, now_iso


_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_filename(s: str) -> str:
    """Replace Windows-illegal characters with ``_`` and trim trailing
    dots / spaces (also illegal as Windows filename suffixes).
    """
    cleaned = _UNSAFE.sub("_", s).rstrip(" .")
    return cleaned or "_"


def materialize_chunk(chunk: Chunk, imports_root: Path) -> Path:
    """Write a single chunk to disk; return the resulting path.

    Idempotent at content level: if the target file already exists with
    byte-identical content, no write occurs.
    """
    target_dir = imports_root / safe_filename(chunk.source_client) / safe_filename(chunk.session_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{safe_filename(chunk.chunk_id)}.md"

    content = _render(chunk)
    if target.exists():
        try:
            if target.read_text(encoding="utf-8") == content:
                return target
        except OSError:
            pass  # fall through to a fresh write
    target.write_text(content, encoding="utf-8", newline="\n")
    return target


def materialize_pending(
    episodic: EpisodicStore,
    imports_root: Path,
    cursor_path: Path,
    *,
    limit: int | None = None,
    save_cursor: bool = True,
) -> list[Path]:
    """Drain pending chunks from ``chunks.jsonl`` to ``.md`` files.

    Reads chunks in stored order via ``episodic.iter_chunks()``. Skips
    everything up to (and including) the chunk id in the cursor; writes
    the rest, stopping after ``limit`` writes when set. The cursor is
    updated to the id of the last chunk written in this pass (or
    unchanged when nothing new arrived).

    ``save_cursor=False`` lets the caller defer the cursor commit until
    downstream work (e.g. qmd indexing) succeeds — use
    :func:`save_pending_cursor` once safe (F-3).

    Returns the list of paths written by this call (empty on no-op).
    """
    state = _load_cursor(cursor_path)
    last_seen = state.get("last_chunk_id")

    skipping = last_seen is not None
    written: list[Path] = []
    new_last: str | None = last_seen

    for chunk in episodic.iter_chunks():
        if skipping:
            if chunk.chunk_id == last_seen:
                skipping = False
            continue
        if limit is not None and len(written) >= limit:
            break
        path = materialize_chunk(chunk, imports_root)
        written.append(path)
        new_last = chunk.chunk_id

    if skipping:
        # Cursor referenced a chunk id we never saw — be loud: leave the
        # cursor alone and surface nothing. Caller can inspect the cursor
        # file directly. We don't raise because chunks.jsonl is allowed
        # to be rewritten / truncated in tests.
        return []

    if save_cursor and written and new_last and new_last != last_seen:
        _save_cursor(cursor_path, new_last)
    return written


def save_pending_cursor(cursor_path: Path, written: list[Path]) -> None:
    """Commit the cursor to the last path in ``written``.

    Caller-driven variant for the F-3 ordering fix: ``materialize_pending``
    is invoked with ``save_cursor=False``; the caller runs downstream work
    (qmd index) and only on success calls this to commit the cursor.
    """
    if not written:
        return
    last_chunk_id = written[-1].stem
    _save_cursor(cursor_path, last_chunk_id)


# -- internals ----------------------------------------------------------


def _render(chunk: Chunk) -> str:
    total_chars = sum(len(t.text) for t in chunk.turns)
    fm_lines = [
        "---",
        f"chunk_id: {chunk.chunk_id}",
        f"session_id: {chunk.session_id}",
        f"source_client: {chunk.source_client}",
        f"started_at: {chunk.started_at}",
        f"ended_at: {chunk.ended_at}",
        f"turn_count: {len(chunk.turns)}",
        f"total_chars: {total_chars}",
    ]
    # CC working dir (when present) — lets the writer resolve a produced-doc
    # basename to a full path (cwd + basename). Omitted for non-CC chunks.
    if chunk.cwd:
        fm_lines.append(f"cwd: {chunk.cwd}")
    # Document-type file paths touched by tool-events this session — full paths
    # for resolving an input/output doc's basename even when it isn't in cwd.
    if chunk.doc_paths:
        import json
        fm_lines.append(f"doc_paths: {json.dumps(chunk.doc_paths, ensure_ascii=False)}")
    fm_lines.append("---")
    body: list[str] = []
    for turn in chunk.turns:
        role = "User" if turn.role == "user" else (
            "Assistant" if turn.role == "assistant" else turn.role.capitalize()
        )
        body.append("")
        body.append(f"## {role} — {turn.timestamp}")
        body.append("")
        body.append(turn.text)
    return "\n".join(fm_lines + body) + "\n"


def _load_cursor(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cursor(path: Path, last_chunk_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"last_chunk_id": last_chunk_id, "updated_at": now_iso()}
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
