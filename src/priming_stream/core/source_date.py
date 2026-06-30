"""Derive a record's ``source_date`` — the real conversation timestamp of
the turn it anchors to.

A record's ``created_at`` is the EXTRACTION timestamp (uniform across a
coldstart — e.g. every W7 record is dated the same day), so it is useless
for recency / supersession reasoning. The real conversation moment IS
recoverable, deterministically and without an LLM, from the record's
``source_uri`` plus its ``anchor_offset_start``:

    source_uri  -> the materialized export .md under ``corpus/imports/``
    anchor      -> nearest ``## User|Assistant — <ts>`` header at-or-before it
    fallback    -> the export frontmatter ``started_at`` (conversation start)

Offsets are CHARACTER offsets, not bytes: the canonical reader
(``graph_ops.records_search`` slices ``text[lo:hi]`` on a ``str``), so the
anchor is a Python string index. We compare against ``match.start()``
directly — using byte positions would drift on diacritic-heavy Romanian
text and select an earlier turn than the one the record anchors to.

Only ``qmd://`` (conversation) records get a ``source_date``. index_cards
(``doc://`` / ``file://``) and owner-authored records have no conversation
date -> ``None``.

SQL-canonical (2026-06-12): records live in SQLite, so the per-cycle
derivation (``ensure_source_dates_in_db``) reads STAGED rows and UPDATEs
them in place — the export ``.md`` files under ``corpus/imports/`` it
derives FROM are the episodic layer and remain on disk unchanged.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from priming_stream.core import source_uri as source_uri_mod

# ``## User — <ts>`` / ``## Assistant — <ts>``. The separator is an em dash
# (U+2014) followed by a space, matching the materialized export format.
_TURN_TS_RE = re.compile(r"^## (?:User|Assistant) — (.+)$", re.MULTILINE)
_STARTED_AT_RE = re.compile(r"^started_at:\s*(.+)$", re.MULTILINE)


def started_at_of(export_text: str) -> str | None:
    """Conversation-level ``started_at`` from an export's frontmatter."""
    m = _STARTED_AT_RE.search(export_text)
    return m.group(1).strip() if m else None


def turn_offsets(export_text: str) -> list[tuple[int, str]]:
    """``(char_offset, timestamp)`` for every turn header, in document order.

    Char offsets match the anchor semantics (the canonical reader slices a
    ``str`` by these offsets), so a record's ``anchor_offset_start`` is
    directly comparable to ``match.start()``.
    """
    return [(m.start(), m.group(1).strip()) for m in _TURN_TS_RE.finditer(export_text)]


def nearest_turn_ts(turns: list[tuple[int, str]], offset: int) -> str | None:
    """Timestamp of the last turn header at-or-before ``offset`` (chars).

    ``turns`` must be in ascending offset order (``turn_offsets`` is).
    Returns ``None`` if no header precedes the offset (caller falls back to
    ``started_at``).
    """
    best: str | None = None
    for pos, ts in turns:
        if pos <= offset:
            best = ts
        else:
            break
    return best


def resolve_source_date(
    source_uri: str,
    anchor_offset_start: int | None,
    *,
    storage_dir: Path,
    corpus_dir: Path,
) -> str | None:
    """Derive ``source_date`` for one record. ``None`` when not a
    conversation record, the export is missing, or no timestamp is
    recoverable. Reads the export file each call — fine for the few records
    per sleep cycle; the staged-row pass caches per-file instead (see
    ``ensure_source_dates_in_db``)."""
    try:
        uri = source_uri_mod.parse(source_uri)
    except ValueError:
        return None
    if uri.scheme != "qmd":
        return None  # file:// / doc:// / owner records carry no conv date
    try:
        path = source_uri_mod.resolve(uri, storage_dir, corpus_dir)
    except ValueError:
        return None
    if path is None or not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    started = started_at_of(text)
    if anchor_offset_start is not None:
        ts = nearest_turn_ts(turn_offsets(text), anchor_offset_start)
        if ts:
            return ts
    return started


# -- staged-row derivation (SQL-canonical) ---------------------------------


def ensure_source_dates_in_db(
    conn: sqlite3.Connection,
    *,
    storage_dir: Path,
    corpus_dir: Path,
) -> dict:
    """Derive ``source_date`` for every STAGED row that lacks one and is a
    resolvable conversation record, UPDATEing the row in place. Idempotent
    (rows that already carry one, or have no conversation date, are left
    untouched). Caches each export file's started_at + turn offsets so a
    multi-record export is read once. Runs in sleep-finalize BEFORE the
    staging→records promotion, so promoted rows carry their date.

    Returns ``{written, skipped_existing, no_date, malformed}`` (the same
    metric keys the old per-``.md`` variant reported; ``malformed`` is
    always 0 — a staged row can't fail to parse).
    """
    metrics = {"written": 0, "skipped_existing": 0, "no_date": 0, "malformed": 0}

    # path -> (started_at, turn_offsets) cache
    export_cache: dict[Path, tuple[str | None, list[tuple[int, str]]]] = {}

    def _export_index(path: Path) -> tuple[str | None, list[tuple[int, str]]]:
        cached = export_cache.get(path)
        if cached is not None:
            return cached
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            idx = (started_at_of(text), turn_offsets(text))
        except OSError:
            idx = (None, [])
        export_cache[path] = idx
        return idx

    rows = conn.execute(
        "SELECT id, source_uri, anchor_offset_start, source_date "
        "FROM records_staging ORDER BY id"
    ).fetchall()
    for row in rows:
        rid, uri_str, anchor, existing = row[0], row[1], row[2], row[3]
        if existing:
            metrics["skipped_existing"] += 1
            continue
        try:
            uri = source_uri_mod.parse(uri_str or "")
        except ValueError:
            metrics["no_date"] += 1
            continue
        if uri.scheme != "qmd":
            metrics["no_date"] += 1
            continue
        try:
            export_path = source_uri_mod.resolve(uri, storage_dir, corpus_dir)
        except ValueError:
            metrics["no_date"] += 1
            continue
        if export_path is None or not export_path.exists():
            metrics["no_date"] += 1
            continue

        started, turns = _export_index(export_path)
        value = (
            nearest_turn_ts(turns, anchor) if anchor is not None else None
        ) or started
        if not value:
            metrics["no_date"] += 1
            continue

        conn.execute(
            "UPDATE records_staging SET source_date = ? WHERE id = ?",
            (value, rid),
        )
        metrics["written"] += 1
    conn.commit()
    return metrics
