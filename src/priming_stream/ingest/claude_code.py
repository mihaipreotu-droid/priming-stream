"""Claude Code ingestion adapter.

Real transcripts live as one ``.jsonl`` file per session under
``~/.claude/projects/<project>/<session-uuid>.jsonl``. Each line is a JSON
object; conversation lines carry ``type`` in {"user","assistant"} and a
``message`` payload. The synthetic fixture mirrors this shape.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from priming_stream.core.models import Chunk, Turn

from .base import Adapter, _parse_ts, split_bursts


def _block_text(content: object) -> str:
    """Extract turn text from a ``message.content`` payload.

    ``user`` content is a plain string; ``assistant`` content is a list of
    text blocks. A plain string for an assistant is also tolerated.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _raw_turn(record: dict) -> tuple[str, str, str, str] | None:
    """Normalize one transcript record to (session_id, role, text, timestamp).

    Returns ``None`` for any record that is not a usable conversation turn.
    """
    if record.get("type") not in ("user", "assistant"):
        return None

    message = record.get("message")
    if not isinstance(message, dict):
        return None

    role = message.get("role") or record.get("type")
    if role not in ("user", "assistant"):
        return None

    timestamp = record.get("timestamp")
    session_id = record.get("sessionId")
    if not isinstance(timestamp, str) or not isinstance(session_id, str):
        return None

    text = _block_text(message.get("content"))
    return session_id, role, text, timestamp


_DOC_EXTS = {
    ".pdf", ".pptx", ".ppt", ".docx", ".doc", ".odt", ".xlsx", ".csv",
    ".rtf", ".epub",
}
_FILE_TOOLS = {"Read", "Write", "Edit", "NotebookEdit", "MultiEdit"}


def _doc_tool_paths(record: dict) -> list[str]:
    """Full paths of document-type files touched by Read/Write/Edit-family
    ``tool_use`` blocks in this record (the blocks ``_block_text`` drops).
    Code/note types (.md/.txt) are excluded to keep the set small — those
    resolve via cwd + basename when they sit in the working dir."""
    message = record.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if not isinstance(content, list):
        return []
    out: list[str] = []
    for block in content:
        if (isinstance(block, dict) and block.get("type") == "tool_use"
                and block.get("name") in _FILE_TOOLS):
            fp = (block.get("input") or {}).get("file_path")
            if isinstance(fp, str) and Path(fp).suffix.lower() in _DOC_EXTS:
                out.append(fp)
    return out


def parse_session_file(
    path: Path,
) -> tuple[list[tuple[str, str, str, str]], str | None, list[str]]:
    """Parse a transcript ``.jsonl`` into ``(raw turns, session cwd, doc paths)``.

    Malformed lines, junk lines, and non-conversation records are skipped;
    a bad line never aborts the parse. ``cwd`` = the first non-empty top-level
    ``cwd`` field (constant per session — the working directory CC ran in).
    ``doc_paths`` = full paths of document-type files touched by file tools this
    session — both for resolving a doc's LOCALPATH basename to a real path.
    """
    raw: list[tuple[str, str, str, str]] = []
    cwd: str | None = None
    doc_paths: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(record, dict):
                continue
            if cwd is None:
                c = record.get("cwd")
                if isinstance(c, str) and c:
                    cwd = c
            doc_paths.update(_doc_tool_paths(record))
            turn = _raw_turn(record)
            if turn is not None:
                raw.append(turn)
    return raw, cwd, sorted(doc_paths)


class _LineJsonAdapter(Adapter):
    """Shared adapter for line-oriented JSON transcripts.

    Subclasses set :attr:`source_client`; the chunk_id prefix derives from it.
    """

    source_client: str = "line_json"

    def __init__(
        self,
        source_path: Path | str,
        idle_minutes: int = 30,
        chunk_max_turns: int = 120,
    ) -> None:
        self.source_path = Path(source_path)
        self.idle_minutes = idle_minutes
        self.chunk_max_turns = chunk_max_turns

    def _session_files(self) -> list[Path]:
        if self.source_path.is_dir():
            return sorted(self.source_path.glob("*.jsonl"))
        if self.source_path.is_file():
            return [self.source_path]
        return []

    def iter_chunks(self) -> Iterator[Chunk]:
        # Collect raw turns grouped by session across all files.
        by_session: dict[str, list[tuple[str, str, str, str]]] = {}
        cwd_by_session: dict[str, str | None] = {}
        docs_by_session: dict[str, list[str]] = {}
        for path in self._session_files():
            rows, file_cwd, file_docs = parse_session_file(path)
            for session_id, role, text, timestamp in rows:
                by_session.setdefault(session_id, []).append(
                    (session_id, role, text, timestamp)
                )
                cwd_by_session.setdefault(session_id, file_cwd)
                docs_by_session.setdefault(session_id, file_docs)

        for session_id in sorted(by_session):
            rows = sorted(by_session[session_id], key=lambda r: _parse_ts(r[3]))
            turns = [
                Turn(index=0, role=role, text=text, timestamp=timestamp)
                for (_, role, text, timestamp) in rows
            ]
            bursts = split_bursts(turns, self.idle_minutes, self.chunk_max_turns)
            for burst_index, burst in enumerate(bursts):
                indexed = [
                    Turn(index=i, role=t.role, text=t.text, timestamp=t.timestamp)
                    for i, t in enumerate(burst)
                ]
                yield Chunk(
                    chunk_id=f"{self.source_client}_{session_id}_{burst_index:03d}",
                    source_client=self.source_client,
                    session_id=session_id,
                    started_at=indexed[0].timestamp,
                    ended_at=indexed[-1].timestamp,
                    turns=indexed,
                    cwd=cwd_by_session.get(session_id),
                    doc_paths=docs_by_session.get(session_id) or [],
                )


class ClaudeCodeAdapter(_LineJsonAdapter):
    """Adapter for Claude Code session transcripts."""

    source_client = "claude_code"
