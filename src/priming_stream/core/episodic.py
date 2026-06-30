"""Episodic store — append-only JSONL files for events, chunks, processing log."""
from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path

from priming_stream.core.models import Chunk, chunk_from_dict, chunk_to_dict, now_iso


class EpisodicStore:
    def __init__(self, episodic_dir: Path | str) -> None:
        self.dir = Path(episodic_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.dir / "live_events.jsonl"
        self.chunks_path = self.dir / "chunks.jsonl"
        self.processed_path = self.dir / "processed.jsonl"
        # Lazy-init cache for known chunk_ids — avoids O(N²) full-scan on each
        # write_chunk call.  None = cache not yet populated.
        self._known_chunk_ids: set[str] | None = None

    def append_event(self, event: dict) -> None:
        self._append(self.events_path, event)

    def write_chunk(self, chunk: Chunk) -> None:
        """Append a chunk, idempotent on ``chunk_id``.

        Re-ingesting the same transcript is a no-op for an already-written
        chunk: a duplicate record would otherwise be replayed by a second
        sleep cycle and double-count its signed edge weights (D30 — weights
        are unbounded).
        """
        if self._known_chunk_ids is None:
            self._known_chunk_ids = self._chunk_ids()
        if chunk.chunk_id in self._known_chunk_ids:
            return
        self._append(self.chunks_path, chunk_to_dict(chunk))
        self._known_chunk_ids.add(chunk.chunk_id)

    def iter_chunks(self) -> Iterator[Chunk]:
        """Yield each stored chunk once; defensively dedupe by ``chunk_id``."""
        seen: set[str] = set()
        for record in self._read(self.chunks_path):
            chunk = chunk_from_dict(record)
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            yield chunk

    def iter_unprocessed_chunks(self) -> Iterator[Chunk]:
        done = self._processed_ids()
        for chunk in self.iter_chunks():
            if chunk.chunk_id not in done:
                yield chunk

    def _chunk_ids(self) -> set[str]:
        return {r.get("chunk_id") for r in self._read(self.chunks_path)}

    def mark_processed(self, chunk_id: str, cycle_id: int) -> None:
        self._append(
            self.processed_path,
            {"chunk_id": chunk_id, "cycle_id": cycle_id, "at": now_iso()},
        )

    def _processed_ids(self) -> set[str]:
        return {r["chunk_id"] for r in self._read(self.processed_path)}

    @staticmethod
    def _append(path: Path, record: dict) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _read(path: Path) -> Iterator[dict]:
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    print(
                        f"episodic: skipping corrupt line {lineno} in {path}",
                        file=sys.stderr,
                    )
