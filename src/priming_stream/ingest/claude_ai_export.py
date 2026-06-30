"""claude.ai data-export ingestion adapter (v0.7-b M4).

claude.ai's "Export your data" button ships a ZIP archive containing
``conversations.json`` — a JSON array of conversation objects. Each
conversation has ``uuid``, ``name``, ``created_at`` and a ``chat_messages``
list; messages carry ``sender`` (``"human"`` or ``"assistant"``), ``text``,
and ``created_at``. This adapter accepts either the archive itself or an
extracted directory and yields one :class:`Chunk` per conversation.

Best-effort shape: when a field is shaped unexpectedly the adapter raises
with the offending field cited rather than silently parsing partial data.
"""
from __future__ import annotations

import json
import zipfile
from collections.abc import Iterator
from pathlib import Path

from priming_stream.core.models import Chunk, Turn

from .base import Adapter, split_bursts

_SENDER_TO_ROLE = {"human": "user", "assistant": "assistant"}


def _read_export_json(path: Path) -> list:
    """Load ``conversations.json`` from a directory or ZIP at ``path``."""
    if path.is_dir():
        target = path / "conversations.json"
        if not target.is_file():
            raise FileNotFoundError(
                f"conversations.json not found in directory {path}"
            )
        try:
            return json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"conversations.json is not valid JSON: {exc}"
            ) from exc

    if path.is_file() and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            target = None
            for name in names:
                if name.endswith("conversations.json"):
                    target = name
                    break
            if target is None:
                raise FileNotFoundError(
                    f"conversations.json not found in ZIP {path}"
                )
            with zf.open(target) as fh:
                try:
                    return json.loads(fh.read().decode("utf-8"))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"conversations.json in {path} is not valid JSON: {exc}"
                    ) from exc

    if not path.exists():
        raise FileNotFoundError(f"export path does not exist: {path}")
    raise RuntimeError(
        f"export path must be a directory or ZIP archive: {path}"
    )


def _conv_to_turns(conv: dict) -> tuple[str, list[Turn]]:
    """Parse one export conversation into (uuid, ordered turns).

    Raises :class:`RuntimeError` if a required field is the wrong shape.
    Returns an empty turn list if the conversation has no usable messages
    — the caller decides what to do (typically skip).
    """
    if not isinstance(conv, dict):
        raise RuntimeError(
            f"conversation entry is not an object: {type(conv).__name__}"
        )

    uuid = conv.get("uuid")
    if not isinstance(uuid, str) or not uuid:
        raise RuntimeError("conversation missing required field 'uuid'")

    messages = conv.get("chat_messages")
    if messages is None:
        raise RuntimeError(
            f"conversation {uuid!r} missing required field 'chat_messages'"
        )
    if not isinstance(messages, list):
        raise RuntimeError(
            f"conversation {uuid!r} field 'chat_messages' must be a list"
        )

    turns: list[Turn] = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise RuntimeError(
                f"conversation {uuid!r} chat_messages[{i}] is not an object"
            )
        sender = msg.get("sender")
        if sender not in _SENDER_TO_ROLE:
            raise RuntimeError(
                f"conversation {uuid!r} chat_messages[{i}] has unexpected "
                f"sender {sender!r}; expected 'human' or 'assistant'"
            )
        text = msg.get("text", "")
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        ts = msg.get("created_at") or conv.get("created_at") or ""
        if not isinstance(ts, str):
            ts = str(ts)
        turns.append(
            Turn(
                index=len(turns),
                role=_SENDER_TO_ROLE[sender],
                text=text,
                timestamp=ts,
            )
        )

    return uuid, turns


def _conv_to_chunks(
    conv: dict,
    *,
    idle_minutes: int,
    chunk_max_turns: int,
    chunk_max_chars: int | None,
) -> list[Chunk]:
    """Build a list of :class:`Chunk`s from one export conversation.

    v0.7-f: applies ``split_bursts`` so a long conversation becomes
    several sub-chunks at idle-gap boundaries, with a turn-count cap
    and a char-budget cap. Single-burst conversations keep the legacy
    ``export_<uuid>`` chunk_id; multi-burst conversations get a
    ``_p{N}`` suffix per piece so the episodic dedup contract stays
    well-defined.
    """
    uuid, turns = _conv_to_turns(conv)
    if not turns:
        return []

    bursts = split_bursts(
        turns,
        idle_minutes=idle_minutes,
        chunk_max_turns=chunk_max_turns,
        chunk_max_chars=chunk_max_chars,
    )

    if len(bursts) == 1:
        # Back-compat: single-burst conversations keep the legacy id.
        burst = bursts[0]
        return [Chunk(
            chunk_id=f"export_{uuid}",
            source_client="claude_ai_export",
            session_id=uuid,
            started_at=burst[0].timestamp,
            ended_at=burst[-1].timestamp,
            turns=burst,
        )]

    chunks: list[Chunk] = []
    for idx, burst in enumerate(bursts):
        # Re-index turn.index within the burst so each chunk has 0-based
        # turns (matches the rest of the pipeline's assumptions).
        local_turns = [
            Turn(index=i, role=t.role, text=t.text, timestamp=t.timestamp)
            for i, t in enumerate(burst)
        ]
        chunks.append(Chunk(
            chunk_id=f"export_{uuid}_p{idx}",
            source_client="claude_ai_export",
            session_id=uuid,
            started_at=local_turns[0].timestamp,
            ended_at=local_turns[-1].timestamp,
            turns=local_turns,
        ))
    return chunks


# Legacy single-chunk builder retained for tests / external callers
# that want the unsplit form. Returns None for empty conversations.
def _conv_to_chunk(conv: dict) -> Chunk | None:
    """Build a single :class:`Chunk` from one export conversation, no split."""
    chunks = _conv_to_chunks(
        conv,
        idle_minutes=10_000_000,   # effectively no idle split
        chunk_max_turns=10_000_000,
        chunk_max_chars=None,
    )
    return chunks[0] if chunks else None


class ClaudeAiExportAdapter(Adapter):
    """Adapter for claude.ai data-export archives.

    v0.7-f: ingestion-time chunk splitting. A single ``conversations.json``
    entry can run to ~900 KB (~300K tokens) — well past the Opus 200K
    context window. ``split_bursts`` (from ``base.py``) is applied with
    idle-gap + turn-count + char-budget caps so each emitted chunk fits
    cleanly through the judge.

    Defaults pin the safe budget (idle_minutes=30, chunk_max_turns=120,
    chunk_max_chars=100_000) so callers that don't thread config through
    still get the protection. Tests historically pass only ``path`` and
    rely on the defaults.
    """

    source_client = "claude_ai_export"

    def __init__(
        self,
        path: Path | str,
        *,
        idle_minutes: int = 30,
        chunk_max_turns: int = 120,
        chunk_max_chars: int | None = 100_000,
    ) -> None:
        self.path = Path(path)
        self.idle_minutes = idle_minutes
        self.chunk_max_turns = chunk_max_turns
        self.chunk_max_chars = chunk_max_chars

    def iter_chunks(self) -> Iterator[Chunk]:
        data = _read_export_json(self.path)
        if not isinstance(data, list):
            raise RuntimeError(
                f"conversations.json in {self.path} must be a JSON array; "
                f"got {type(data).__name__}"
            )
        for conv in data:
            for chunk in _conv_to_chunks(
                conv,
                idle_minutes=self.idle_minutes,
                chunk_max_turns=self.chunk_max_turns,
                chunk_max_chars=self.chunk_max_chars,
            ):
                yield chunk
