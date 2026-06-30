"""Heuristic candidate filter for record extraction (v0.7-x W-C).

The sleep cycle calls ``score(chunk_text, markers)`` to decide whether a
chunk is worth sending to the LLM for record extraction. Markers come
from ``config/record_markers.toml`` (RO + EN, grouped by intent). A chunk
passes if ``score >= THRESHOLD`` (default 0.3 — one hit suffices).

Conventions §11: single owner of marker loading. Both the skill agent
(via Python one-liner) and any CLI helper import from here — no
duplicate parsing.
"""
from __future__ import annotations

import tomllib
from pathlib import Path


THRESHOLD: float = 0.3


def load_markers(path: Path) -> dict[str, dict[str, list[str]]]:
    """Load ``record_markers.toml`` into a nested dict.

    Shape: ``{lang: {intent: [marker, ...], ...}, ...}``. The TOML parser
    already produces this shape; we only enforce types so callers can
    iterate without defensive checks.
    """
    with Path(path).open("rb") as fh:
        raw = tomllib.load(fh)
    out: dict[str, dict[str, list[str]]] = {}
    for lang, intents in raw.items():
        if not isinstance(intents, dict):
            continue
        bucket: dict[str, list[str]] = {}
        for intent, markers in intents.items():
            if isinstance(markers, list):
                bucket[intent] = [str(m) for m in markers]
        out[lang] = bucket
    return out


def score(
    text: str,
    markers: dict[str, dict[str, list[str]]],
) -> tuple[float, dict[str, list[str]]]:
    """Score ``text`` by counting marker substring hits.

    Returns ``(score, hits)`` where:

    - ``score = min(1.0, hit_count * 0.3)`` — one hit lands at the
      default threshold, three hits saturate.
    - ``hits`` groups matched marker strings by intent (collapsing across
      languages, since ``decision`` in RO and EN target the same concept
      downstream). A marker that hits multiple times still counts once
      toward the hits dict for that intent but ``hit_count`` for scoring
      counts each (intent, marker) pair once — same marker matched in two
      languages = two hits.
    """
    if not text:
        return 0.0, {}
    haystack = text.lower()
    hits: dict[str, list[str]] = {}
    hit_count = 0
    for _lang, intents in markers.items():
        for intent, marker_list in intents.items():
            for marker in marker_list:
                if not marker:
                    continue
                if marker.lower() in haystack:
                    hit_count += 1
                    bucket = hits.setdefault(intent, [])
                    if marker not in bucket:
                        bucket.append(marker)
    s = min(1.0, hit_count * 0.3)
    return s, hits
