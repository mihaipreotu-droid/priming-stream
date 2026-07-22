"""Shared bridge interface types (v0.7-x Component A — read-time restructuring).

The frozen interface contract between the Component-A modules, made
executable. The two-seed walk (``spreading.walk_two_seeds``), recency
selection (``recency.select_semantic``), the lexical bucket
(``lexical.lexical_bucket``), and the orchestrator
(``working_set.build_priming``) all speak these types.

``ScoredRecord`` carries a :class:`~priming_stream.core.models.Record` plus a
single float:

* bucket A (semantic): the recency-weighted ``rank_score`` after selection,
  or the raw combined ``max(act_prompt, act_response)`` before selection.
* bucket B (lexical): the raw BM25 value (lower = better match); ordering is
  carried by list position, not by this field — render ignores it.

Pure data only: no logic, no I/O, no imports beyond the Record model. This
file is a contract artifact (single owner = the panel baseline); modules
import from it but never redefine it.
"""
from __future__ import annotations

from dataclasses import dataclass

from priming_stream.core.models import Record


@dataclass(frozen=True)
class ScoredRecord:
    """A record paired with one score. See module docstring for the
    score's meaning per bucket."""
    record: Record
    score: float


@dataclass(frozen=True)
class PrimingResult:
    """The two priming buckets produced by ``build_priming``.

    ``semantic`` (bucket A) is ranked descending by recency-weighted score
    and truncated to the semantic budget (``bucket_total - bucket_lexical``).
    ``lexical`` (bucket B) is BM25-ordered, A-first-deduped, and truncated to
    ``bucket_lexical``. Either may be empty.
    """
    semantic: list[ScoredRecord]
    lexical: list[ScoredRecord]
    # P2/P3 turn-gate provenance (2026-07-21, final amendment: full or
    # whisper, never silence). "full" (no gating / gate off / kickoff),
    # "whisper-floor" (turn top under cfg.turn_floor — rendered with the
    # weak-field marker), "whisper-regime" (tool-dense turn), or
    # "whisper-notification" (<task-notification> turn). Whisper = top
    # cfg.whisper_k semantic + top cfg.whisper_lex_k lexical.
    gated: str = "full"
