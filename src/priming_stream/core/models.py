"""Core data models and serialization helpers (v0.7-x).

Records-as-substrate: Record carries just a summary anchored to a source
URI + offsets. Chunks and Turns are the episodic input. Nodes/Edges/
Decisions are gone in v0.7-x.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_record_id() -> str:
    """Record id = 'rec_' + first 8 hex chars of a fresh UUIDv4.

    ~1.1bn space; collision-check is the caller's job (filesystem +
    SQLite PRIMARY KEY both raise on collision). Sized for the POC.
    """
    return "rec_" + uuid.uuid4().hex[:8]


@dataclass
class Turn:
    index: int
    role: str  # "user" | "assistant"
    text: str
    timestamp: str


@dataclass
class Chunk:
    chunk_id: str
    source_client: str
    session_id: str
    started_at: str
    ended_at: str
    turns: list[Turn] = field(default_factory=list)
    # Claude Code working directory (top-level ``cwd`` on each transcript
    # record). Carried so produced/processed documents — whose full path is
    # NOT in the conversation text (only the basename is; the path lives in a
    # stripped tool_use / is skill-produced) — can be resolved at write time as
    # cwd + basename. None for non-CC sources (claude.ai exports have no cwd).
    cwd: str | None = None
    # Full paths of document-type files touched by Read/Write/Edit/NotebookEdit
    # tool-events this session (the tool_use blocks the adapter otherwise drops).
    # Lets the writer resolve an INPUT or OUTPUT doc's LOCALPATH basename to a
    # real path even when the file is NOT in cwd (a client PDF in Downloads, an
    # output written elsewhere). Empty for non-CC sources.
    doc_paths: list[str] = field(default_factory=list)


@dataclass
class Record:
    id: str                              # 'rec_' + 8 hex
    source_uri: str                      # 'qmd://...' or 'file:///...'
    anchor_offset_start: int | None
    anchor_offset_end: int | None
    summary: str                         # 1-3 sentences, plain text
    created_at: str                      # ISO 8601 UTC with 'Z' (EXTRACTION date)
    # v0.7-x-piece3 (document ingestion). These four are OPTIONAL and only
    # populated for ``kind == 'index_card'`` records. A claim record leaves
    # them at their defaults (kind='claim', the rest None) — no caller may
    # require doc_key/source/content_hash on a claim. Defaults appended at
    # the END of the field list so every existing positional/kwarg
    # construction of Record keeps working unchanged.
    kind: str = "claim"                  # 'claim' | 'index_card'
    doc_key: str | None = None           # index_card: canonical identity; claim: doc reference
    source: str | None = None            # index_card: disk path / URL / None
    content_hash: str | None = None      # index_card: source change-detection
    title: str | None = None             # document title (card, or claim referencing a doc)
    provisional: bool = False            # index_card: True = stub (no file yet, unverified)
    # v0.7-x-B (sleep hygiene): real conversation timestamp of the anchored
    # turn (date+time, ISO). created_at is the uniform extraction date; this
    # is the recency/supersession key. Derived from source_uri+anchor
    # (core/source_date.py). None for index_cards / owner-authored records.
    source_date: str | None = None


def chunk_to_dict(chunk: Chunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "source_client": chunk.source_client,
        "session_id": chunk.session_id,
        "started_at": chunk.started_at,
        "ended_at": chunk.ended_at,
        "cwd": chunk.cwd,
        "doc_paths": chunk.doc_paths,
        "turns": [
            {
                "index": t.index,
                "role": t.role,
                "text": t.text,
                "timestamp": t.timestamp,
            }
            for t in chunk.turns
        ],
    }


def chunk_from_dict(d: dict) -> Chunk:
    return Chunk(
        chunk_id=d["chunk_id"],
        source_client=d["source_client"],
        session_id=d["session_id"],
        started_at=d["started_at"],
        ended_at=d["ended_at"],
        cwd=d.get("cwd"),
        doc_paths=d.get("doc_paths") or [],
        turns=[
            Turn(
                index=t["index"],
                role=t["role"],
                text=t["text"],
                timestamp=t["timestamp"],
            )
            for t in d.get("turns", [])
        ],
    )
