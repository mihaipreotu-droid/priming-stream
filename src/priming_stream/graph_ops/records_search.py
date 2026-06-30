"""Records ops for the MCP tool surface (v0.7-x-vec-index).

Three functions, all JSON-compatible return shapes:

- :func:`graph_search_records` — vec_index search on the records collection,
  returns enriched ``{record_id, summary, score, source_uri}`` hits.
- :func:`graph_records` — single-record lookup by id (None when unknown).
- :func:`graph_chunk_around_anchor` — read the source file the record
  anchors to and return a slice ±window around the anchor. Implements the
  §16.6 "records prime; chunks verify" verification path.

The vec_index returns record IDs directly via :class:`VecHit.record_id`,
so this module no longer parses IDs out of qmd-style paths. Hits whose
id isn't in SQLite are dropped — the vector index can outrun the SQLite
mirror in time, but a stale hit has no metadata to enrich with.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from priming_stream.core.graph_repo import GraphRepo
from priming_stream.core.models import Record
from priming_stream.core import source_uri as source_uri_mod
from priming_stream.integrations.vec_index import RecordsVecIndex


def _record_to_dict(record: Record) -> dict:
    return {
        "id": record.id,
        "source_uri": record.source_uri,
        "anchor_offset_start": record.anchor_offset_start,
        "anchor_offset_end": record.anchor_offset_end,
        "summary": record.summary,
        "created_at": record.created_at,
    }


def graph_search_records(
    query_text: str,
    k: int,
    vec_index: RecordsVecIndex,
    repo: GraphRepo,
) -> list[dict]:
    """Vector search on the records collection, enriched with SQLite metadata.

    Returns ``[{record_id, summary, score, source_uri}, ...]`` ordered by
    descending score. Hits whose id isn't in the SQLite mirror are
    silently skipped — the index can outrun the mirror in time and the
    mirror is the source of truth for record metadata.
    """
    if not query_text:
        return []
    hits = vec_index.search(query_text, k=k)
    out: list[dict] = []
    for hit in hits:
        rid = hit.record_id
        if not rid:
            continue
        record = repo.get_record(rid)
        if record is None:
            continue
        out.append({
            "record_id": rid,
            "summary": record.summary,
            "score": hit.score,
            "source_uri": record.source_uri,
            "source_date": record.source_date,
            "kind": record.kind,
        })
    return out


_LEX_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_LEX_MODES = ("and", "or", "phrase")


def _fts5_match(query_text: str, mode: str) -> str:
    """Build an FTS5 ``MATCH`` expression from free text.

    Tokenize on Unicode word chars, drop single-char tokens (bm25 noise),
    double-quote each token so FTS5 special syntax can't parse-error. ``mode``
    controls combination:

    - ``and`` (default) — every token must appear (precision; "find records
      about X and Y").
    - ``or`` — any token may appear (recall).
    - ``phrase`` — the tokens as one exact ordered phrase ``"t1 t2 ..."``.

    Empty / tokenless input → ``""`` (caller returns no hits).
    """
    tokens = [t for t in _LEX_TOKEN_RE.findall(query_text or "") if len(t) >= 2]
    if not tokens:
        return ""
    if mode == "phrase":
        return '"' + " ".join(tokens) + '"'
    joiner = " OR " if mode == "or" else " AND "
    return joiner.join(f'"{t}"' for t in tokens)


def graph_search_lexical(
    query_text: str,
    k: int,
    mode: str,
    repo: GraphRepo,
) -> list[dict]:
    """On-demand FTS5 BM25 lexical search over record summaries.

    The keyword/term counterpart of :func:`graph_search_records` (semantic):
    use it to find a record by an EXACT term, name, or citation that dense
    embedding similarity would miss (the register-mismatch / bare-citation
    case). Distinct from the bridge's automatic bucket B — here the CALLER
    composes the query (the search intent / mechanism terms, not a whole
    prompt) and picks ``mode``.

    Returns ``[{record_id, summary, score, source_uri, source_date, kind}]``
    ordered BEST-FIRST. ``score`` is the raw SQLite ``bm25`` value (more
    negative = better match); ordering is carried by list position. ``mode``
    is one of ``and`` (default), ``or``, ``phrase``; an unknown mode falls back
    to ``and``. NEVER raises — ``[]`` on missing FTS5 table, empty query, or
    no matches.
    """
    if not query_text:
        return []
    match = _fts5_match(query_text, mode if mode in _LEX_MODES else "and")
    if not match:
        return []
    try:
        rows = repo.conn.execute(
            "SELECT r.id, bm25(records_fts) AS rank "
            "FROM records r JOIN records_fts f ON r.rowid = f.rowid "
            "WHERE f.summary MATCH ? "
            "ORDER BY bm25(records_fts) "
            "LIMIT ?",
            (match, max(1, int(k))),
        ).fetchall()
    except sqlite3.OperationalError:
        return []  # no FTS5 table / malformed MATCH — degrade to empty
    out: list[dict] = []
    for row in rows:
        rid = row[0]
        record = repo.get_record(rid)
        if record is None:
            continue
        out.append({
            "record_id": rid,
            "summary": record.summary,
            "score": float(row[1]),
            "source_uri": record.source_uri,
            "source_date": record.source_date,
            "kind": record.kind,
        })
    return out


def graph_records(record_id: str, repo: GraphRepo) -> dict | None:
    """Return the full record dict by id, or ``None`` if unknown."""
    if not record_id:
        return None
    record = repo.get_record(record_id)
    return _record_to_dict(record) if record is not None else None


def graph_chunk_around_anchor(
    record_id: str,
    window: int,
    repo: GraphRepo,
    storage_dir: Path,
    corpus_dir: Path,
) -> dict:
    """Read the source file for ``record_id`` and return a window of text.

    If the record has both ``anchor_offset_start`` and ``anchor_offset_end``,
    returns ``text[max(0, anchor_start-window):anchor_end+window]``. If
    either anchor is ``None`` (chunk-level record), returns the entire
    body. Unknown record id surfaces as ``{"error": "..."}``.

    ``corpus_dir`` is the post-rename ``storage/corpus/`` root; the
    ``qmd://`` source_uri scheme name is preserved on existing records.
    """
    if not record_id:
        return {"error": "record_id is required"}
    record = repo.get_record(record_id)
    if record is None:
        return {"error": f"record not found: {record_id}"}

    # Index cards (piece3): the card body IS the content — there is no
    # deeper readable "chunk" in the substrate, and ``source_uri`` points
    # at the ORIGINAL document (often a PDF / binary) which this op cannot
    # read as text. Without this guard ``read_text`` below raises an
    # uncaught UnicodeDecodeError on a binary source. Treat the card like a
    # self-anchored record: return its body, plus the ``source`` link so the
    # caller can open the original document directly to verify in depth.
    if getattr(record, "kind", "claim") == "index_card":
        return {
            "record_id": record_id,
            "source_uri": record.source_uri,
            "text": record.summary,
            "is_full_file": False,
            "index_card": True,
            "source": record.source,
        }

    try:
        uri = source_uri_mod.parse(record.source_uri)
    except ValueError as exc:
        return {"error": f"unparseable source_uri: {exc}"}
    try:
        path = source_uri_mod.resolve(uri, storage_dir, corpus_dir)
    except ValueError as exc:
        return {"error": f"cannot resolve source_uri: {exc}"}

    # Self-anchored records (slash commands): no on-disk source. The
    # summary IS the content.
    if path is None:
        return {
            "record_id": record_id,
            "source_uri": record.source_uri,
            "text": record.summary,
            "is_full_file": False,
            "self_anchored": True,
        }

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return {"error": f"cannot read source file: {exc}"}

    start = record.anchor_offset_start
    end = record.anchor_offset_end
    if start is None or end is None:
        slice_text = text
        is_full = True
    else:
        lo = max(0, start - window)
        hi = end + window
        slice_text = text[lo:hi]
        is_full = False

    return {
        "record_id": record_id,
        "source_uri": record.source_uri,
        "text": slice_text,
        "is_full_file": is_full,
    }
