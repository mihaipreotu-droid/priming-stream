"""Ingestion adapter base class and shared burst-splitting helper."""
from __future__ import annotations

import abc
from collections.abc import Iterator
from datetime import datetime, timedelta

from priming_stream.core.models import Chunk, Turn


def _parse_ts(ts: str) -> datetime:
    """Parse an ISO8601 UTC timestamp; tolerate a trailing 'Z'."""
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def split_bursts(
    turns: list[Turn],
    idle_minutes: int,
    chunk_max_turns: int = 120,
    chunk_max_chars: int | None = None,
) -> list[list[Turn]]:
    """Split a time-ordered list of turns into bursts.

    A burst boundary falls wherever two consecutive turns are *more than*
    ``idle_minutes`` apart. Any burst longer than ``chunk_max_turns`` is then
    sub-split into fixed-size pieces so chunks never grow unbounded.

    v0.7-f: if ``chunk_max_chars`` is set, a third pass splits any sub-burst
    whose total turn-text length exceeds the budget. Split points are turn
    boundaries — a single turn is never sliced, even if its own text
    exceeds the budget (in which case that one-turn burst is yielded
    intact and the judge handles the LLM-context-overflow boundary).
    This is the safety net for claude.ai exports where a single
    conversation can run hundreds of thousands of characters across a
    single uninterrupted burst.
    """
    if not turns:
        return []

    idle = timedelta(minutes=idle_minutes)
    bursts: list[list[Turn]] = []
    current: list[Turn] = [turns[0]]

    for prev, turn in zip(turns, turns[1:]):
        if _parse_ts(turn.timestamp) - _parse_ts(prev.timestamp) > idle:
            bursts.append(current)
            current = [turn]
        else:
            current.append(turn)
    bursts.append(current)

    if chunk_max_turns and chunk_max_turns > 0:
        capped: list[list[Turn]] = []
        for burst in bursts:
            for i in range(0, len(burst), chunk_max_turns):
                capped.append(burst[i : i + chunk_max_turns])
        bursts = capped

    if chunk_max_chars and chunk_max_chars > 0:
        char_capped: list[list[Turn]] = []
        for burst in bursts:
            piece: list[Turn] = []
            piece_chars = 0
            for turn in burst:
                t_chars = len(turn.text)
                # Always include the first turn of an empty piece, even
                # if its text already exceeds the budget — never slice a
                # single turn.
                if piece and piece_chars + t_chars > chunk_max_chars:
                    char_capped.append(piece)
                    piece = [turn]
                    piece_chars = t_chars
                else:
                    piece.append(turn)
                    piece_chars += t_chars
            if piece:
                char_capped.append(piece)
        bursts = char_capped

    return bursts


class Adapter(abc.ABC):
    """A source-specific adapter that yields normalized chunks."""

    @abc.abstractmethod
    def iter_chunks(self) -> Iterator[Chunk]:
        """Yield one :class:`Chunk` per interaction burst."""
        raise NotImplementedError
