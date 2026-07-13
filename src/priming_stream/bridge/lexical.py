"""Bucket B — the lexical citation channel (v0.7-x Component A, A.2).

A parallel FTS5 BM25 signal over ``records.summary``, run on the USER
PROMPT ONLY (never the prev-response). Promotes the ``records_fts`` table
from cold-path fallback to a first-class bucket so naming a paper or term
surfaces its record even when dense-embedding spreading misses a bare
citation (the Collins & Loftus case: a query naming the paper must surface
its ``index_card``).

Differs from ``daemon.fallback_lexical.search`` (which is layered ABOVE
the bridge — we do not import from it):

* returns full :class:`~priming_stream.core.models.Record` objects (carrying
  ``source_date`` / ``kind`` for render + kind-bias), not ``(id, summary)``;
* takes an already-open connection (no DB-path / read-only-URI handling);
* applies A-first dedup against ``exclude_ids`` (anti-redundancy with
  bucket A — NOT a relevance filter);
* biases ``index_card`` hits ahead of ``claim`` hits.

The sanitizer is REPLICATED here rather than imported — the daemon is the
higher layer; the bridge must not depend upward on it. NEVER raises.
"""
from __future__ import annotations

import re
import sqlite3

from priming_stream.bridge.types import ScoredRecord
from priming_stream.core.graph_repo import GraphRepo

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _sanitize_for_fts5(query_text: str) -> str:
    """Tokenize + double-quote each token; join with explicit FTS5 ``OR``.

    Word tokens via a Unicode ``\\w+`` regex; single-char tokens dropped
    (BM25 noise); each remaining token wrapped in double quotes so FTS5
    special syntax (``"``, ``*``, ``:``, parens, NEAR/AND/OR/NOT) can't
    parse-error the MATCH. Empty string for empty / tokenless input — the
    caller skips the SQL entirely in that case.

    Tokens are OR-joined, NOT whitespace-joined (FTS5 implicit-AND). A.2
    mandates down-rank, not exclude: a record must surface on ANY prompt
    token, with BM25/IDF down-ranking common terms — intake-cutting is the
    asymmetric error. Implicit-AND would require EVERY prompt token in the
    summary, so a real sentence ("ce zice collins & loftus...") naming a
    paper returns zero. OR lets the named term surface; budget cuts the tail.

    Deliberately DIVERGES from ``daemon.fallback_lexical._sanitize_for_fts5``,
    which keeps implicit-AND — that is a degraded cold-path precise-lexical
    fallback, outside A.2's parallel-signal scope. Kept separate (not hoisted)
    to preserve the bridge→daemon layering (bridge is lower).
    """
    tokens = [t for t in _TOKEN_RE.findall(query_text or "") if len(t) >= 2]
    if not tokens:
        return ""
    return " OR ".join(f'"{t}"' for t in tokens)


def lexical_bucket(
    conn: sqlite3.Connection,
    prompt: str,
    *,
    limit: int,
    exclude_ids: set[str],
    kind_bias: bool = True,
    exclude_recent_ids: frozenset[str] | set[str] = frozenset(),
) -> list[ScoredRecord]:
    """FTS5 BM25 lexical bucket over ``records.summary`` for the PROMPT ONLY.

    Returns up to ``limit`` :class:`ScoredRecord` in surface order. NEVER
    raises — ``[]`` on missing FTS5 table, empty prompt, no matches, or any
    error. ``score`` is the raw bm25 value (lower = better match; render
    ignores it — ordering is carried by list position).

    See the module docstring + the frozen interface contract for the full
    behaviour (sanitize, no threshold, A-first dedup, kind-bias).

    ``exclude_recent_ids`` (item 3.3 cross-turn dedup): ids primed in the last
    N turns of the same session. Folded into the drop set alongside the A-first
    ``exclude_ids``; the over-fetch grows with it so backfill still fills
    ``limit`` from the next BM25 hits. Empty default → no-op.
    """
    try:
        if limit <= 0:
            return []
        match = _sanitize_for_fts5(prompt)
        if not match:
            return []

        # A-first dedup ids + cross-turn recent ids share identical filter
        # behaviour here (drop the row, keep scanning), so fold them into one
        # drop set for the row check.
        drop_ids = set(exclude_ids) | set(exclude_recent_ids)

        # Size the over-fetch on the A-dedup set ONLY — deliberately NOT on
        # ``exclude_recent_ids``. The recent set is large (~all ids primed in
        # the last N turns) and mostly OFF-TOPIC for this prompt, so counting
        # it here would balloon ``fetch_n`` and let ``kind_bias`` promote
        # index_cards that only exist in the enlarged window ahead of genuine
        # top hits — perturbing NON-recent records (item 3.3 turn-69 finding).
        # The ``+ limit`` headroom still absorbs recent hits that DO land in
        # the window (the common repeat case); if more than that are recent,
        # the bucket yields fewer — correct (those were re-shown repeats),
        # never padded. bm25 is raw (lower = better); read positionally.
        fetch_n = limit + len(exclude_ids) + limit
        rows = conn.execute(
            "SELECT r.id, bm25(records_fts) AS rank "
            "FROM records r "
            "JOIN records_fts f ON r.rowid = f.rowid "
            "WHERE f.summary MATCH ? "
            "ORDER BY bm25(records_fts) "
            "LIMIT ?",
            (match, int(fetch_n)),
        ).fetchall()

        # ``GraphRepo._record`` reads columns by name (sqlite3.Row).  Use a
        # cursor-level row_factory so the shared daemon connection is never
        # mutated — avoids a race with concurrent requests in
        # ThreadingHTTPServer.  Python 3.12+ supports cursor.row_factory.
        scored: list[ScoredRecord] = []
        for row in rows:
            rec_id, bm25_val = row[0], row[1]
            if rec_id in drop_ids:
                continue  # A-first dedup + 3.3 cross-turn dedup, not a relevance cut
            cur = conn.cursor()
            cur.row_factory = sqlite3.Row
            db_row = cur.execute(
                "SELECT * FROM records WHERE id = ?", (rec_id,)
            ).fetchone()
            if db_row is None:
                continue
            rec = GraphRepo._record(db_row)
            scored.append(ScoredRecord(record=rec, score=float(bm25_val)))

        if kind_bias:
            # STABLE sort: index_card hits ahead of claims, BM25 order
            # preserved within each kind. Cards are rare in lexical hits;
            # this surfaces a named paper's card (the Collins behaviour).
            scored.sort(key=lambda sr: 0 if sr.record.kind == "index_card" else 1)

        return scored[:limit]
    except Exception:
        return []
