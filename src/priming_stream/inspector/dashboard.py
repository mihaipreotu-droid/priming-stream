"""Static-HTML inspector dashboard (v0.7-x-vec-index).

Records-as-substrate surface: the active graph is now just ``records`` +
``sleep_cycles``, the ``storage/corpus/`` tree holds the chunk/record
.md files, and the ChromaDB ``records`` collection at
``storage/vec_index/chroma/`` is the bridge's search index. The dashboard
renders four panels:

  1. Corpus health      — record total + kind breakdown (claim / index_card),
                          provisional count, source_date span, ChromaDB count,
                          and a SQLite↔Chroma parity hint. Tolerates vec_index
                          init / count errors (renders ``unavailable``).
  2. Records browser    — latest 100 records (id, summary, source_uri,
                          anchor offsets, created_at).
  3. Sleep cycles       — latest 20 audit rows.
  4. Recent bridge      — latest 20 priming calls from ``echoes.jsonl`` (E.1):
     invocations          timestamp, session, source, spread_ms, A/B counts,
                          prompt head, a sample of primed summaries. Absent
                          file → graceful empty note.

The database is opened read-only — the inspector never writes the graph.
"""
from __future__ import annotations

import html
import json
import sqlite3
from pathlib import Path

from priming_stream.core.config import load_config
from priming_stream.core.graph_repo import GraphRepo
from priming_stream.core.paths import resolve_paths
from priming_stream.core.usage_join import (
    attach_usage_to_echoes,
    classify_usage,
    read_usage,
)
from priming_stream.integrations.vec_index import RecordsVecIndex

_SUMMARY_TRUNC = 200
_RECORDS_LIMIT = 100
_CYCLES_LIMIT = 20
_ECHOES_LIMIT = 20


def _connect_readonly(graph_db: Path) -> sqlite3.Connection:
    """Open the graph DB in read-only mode via a URI connection."""
    uri = f"file:{graph_db.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _esc(value: object) -> str:
    """HTML-escape any value, mapping None to an em dash."""
    if value is None:
        return "&mdash;"
    return html.escape(str(value))


def _truncate(text: str, n: int = _SUMMARY_TRUNC) -> str:
    if text is None:
        return ""
    if len(text) <= n:
        return text
    return text[: n - 1] + "…"


_CSS = """
body { font-family: Segoe UI, Arial, sans-serif; margin: 2rem; color: #1c1c1c;
       background: #fafafa; }
h1 { font-size: 1.5rem; } h2 { font-size: 1.1rem; margin-top: 2rem; }
table { border-collapse: collapse; width: 100%; background: #fff;
        margin-top: 0.5rem; }
th, td { border: 1px solid #e0e0e0; padding: 0.35rem 0.6rem;
         text-align: left; font-size: 0.85rem; vertical-align: top; }
th { background: #f0f0f0; }
td.summary { max-width: 40rem; }
td.uri { font-family: Consolas, Menlo, monospace; font-size: 0.78rem;
         word-break: break-all; }
.empty { color: #888; font-style: italic; margin-top: 0.5rem; }
.unavailable { color: #b8860b; font-style: italic; }
"""


def _records_table(repo: GraphRepo) -> str:
    records = repo.list_records(limit=_RECORDS_LIMIT)
    if not records:
        return (
            "<h2>Records</h2>"
            '<p class="empty">No records yet.</p>'
        )
    rows = []
    for r in records:
        rows.append(
            f"<tr><td>{_esc(r.id)}</td>"
            f'<td class="summary">{_esc(_truncate(r.summary))}</td>'
            f'<td class="uri">{_esc(r.source_uri)}</td>'
            f"<td>{_esc(r.anchor_offset_start)}</td>"
            f"<td>{_esc(r.anchor_offset_end)}</td>"
            f"<td>{_esc(r.created_at)}</td></tr>"
        )
    body = "".join(rows)
    return (
        "<h2>Records</h2>"
        f"<p>Showing latest {len(records)} (most recent first).</p>"
        "<table><tr>"
        "<th>id</th><th>summary</th><th>source_uri</th>"
        "<th>anchor start</th><th>anchor end</th><th>created_at</th>"
        f"</tr>{body}</table>"
    )


def _cycles_table(repo: GraphRepo) -> str:
    cycles = repo.list_sleep_cycles(limit=_CYCLES_LIMIT)
    if not cycles:
        return (
            "<h2>Sleep cycles</h2>"
            '<p class="empty">No sleep cycles yet.</p>'
        )
    rows = []
    for c in cycles:
        rows.append(
            f"<tr><td>{_esc(c['id'])}</td>"
            f"<td>{_esc(c['started_at'])}</td>"
            f"<td>{_esc(c['completed_at'])}</td>"
            f"<td>{_esc(c['chunks_materialized'])}</td>"
            f"<td>{_esc(c['records_created'])}</td>"
            f"<td>{_esc(c['records_skipped'])}</td>"
            f"<td>{_esc(c['notes'])}</td></tr>"
        )
    body = "".join(rows)
    return (
        "<h2>Sleep cycles</h2>"
        f"<p>Showing latest {len(cycles)} (most recent first).</p>"
        "<table><tr>"
        "<th>id</th><th>started_at</th><th>completed_at</th>"
        "<th>chunks materialized</th><th>records created</th>"
        "<th>records skipped</th><th>notes</th>"
        f"</tr>{body}</table>"
    )


def _vec_count(vec_index: RecordsVecIndex | None) -> int | None:
    """ChromaDB record count, or None when the index is unavailable."""
    if vec_index is None:
        return None
    try:
        return vec_index.count()
    except Exception:  # noqa: BLE001 - degrade
        return None


def _corpus_health(conn: sqlite3.Connection) -> dict:
    """Aggregate corpus-health counts over the read-only connection.

    Pure SELECTs (no GraphRepo write-surface) — kind breakdown, provisional
    count, and the dated-record ``source_date`` span. Soft-deleted records are
    REMOVED from SQLite (claim_reconcile trashes the row), so there is nothing
    to count for them here — the table is what the bridge actually sees.
    """
    total = conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    by_kind = {
        str(r[0]): r[1]
        for r in conn.execute(
            "SELECT kind, COUNT(*) FROM records GROUP BY kind"
        ).fetchall()
    }
    provisional = conn.execute(
        "SELECT COUNT(*) FROM records WHERE provisional = 1"
    ).fetchone()[0]
    span = conn.execute(
        "SELECT MIN(source_date), MAX(source_date) FROM records "
        "WHERE source_date IS NOT NULL AND source_date != ''"
    ).fetchone()
    return {
        "total": total,
        "by_kind": by_kind,
        "provisional": provisional,
        "date_min": span[0] if span else None,
        "date_max": span[1] if span else None,
    }


def _health_section(
    conn: sqlite3.Connection,
    vec_index: RecordsVecIndex | None,
    persist_dir: Path,
) -> str:
    """Render the corpus-health panel: SQLite aggregates + the ChromaDB count,
    plus a parity hint (SQLite total vs Chroma count — they should match; a gap
    flags a stale index needing ``vec-index-rebuild``)."""
    h = _corpus_health(conn)
    vec = _vec_count(vec_index)
    vec_cell = _esc(vec) if vec is not None else '<span class="unavailable">unavailable</span>'

    claims = h["by_kind"].get("claim", 0)
    cards = h["by_kind"].get("index_card", 0)
    # Any unexpected kinds beyond the two known ones, surfaced honestly.
    other = h["total"] - claims - cards
    if h["date_min"] or h["date_max"]:
        span = f"{_esc(h['date_min'])} &rarr; {_esc(h['date_max'])}"
    else:
        span = "&mdash;"
    parity = (
        "match" if vec is not None and vec == h["total"]
        else (f"&Delta; {h['total'] - vec}" if vec is not None else "&mdash;")
    )

    rows = (
        f"<tr><th>records (SQLite)</th><td>{_esc(h['total'])}</td></tr>"
        f"<tr><th>&nbsp;&nbsp;claims</th><td>{_esc(claims)}</td></tr>"
        f"<tr><th>&nbsp;&nbsp;index_cards</th><td>{_esc(cards)}</td></tr>"
        + (f"<tr><th>&nbsp;&nbsp;other kinds</th><td>{_esc(other)}</td></tr>"
           if other else "")
        + f"<tr><th>provisional</th><td>{_esc(h['provisional'])}</td></tr>"
        f"<tr><th>source_date span</th><td>{span}</td></tr>"
        f"<tr><th>vec_index (Chroma)</th><td>{vec_cell}</td></tr>"
        f"<tr><th>SQLite&harr;Chroma parity</th><td>{parity}</td></tr>"
        f'<tr><th>persist dir</th><td class="uri">{_esc(persist_dir)}</td></tr>'
    )
    return (
        "<h2>Corpus health</h2>"
        f"<table>{rows}</table>"
    )


def _read_echoes(path: Path, limit: int) -> list[dict]:
    """Tolerant tail reader for ``echoes.jsonl`` (blank/corrupt lines skipped,
    same posture as the episodic log + ``cli/echoes.py``). Returns the latest
    ``limit`` dict entries, most recent last."""
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                out.append(rec)
    return out[-max(limit, 1):]


def _used_cell(echo: dict) -> str:
    """Compact per-turn active-use summary (usage.jsonl joined to this echo).

    Renders a tag histogram like ``2×verified-use, 1×recall-miss`` from the
    usage entries attributed to this turn, or ``&mdash;`` when none."""
    used = echo.get("used") or []
    if not used:
        return "&mdash;"
    counts: dict[str, int] = {}
    for u in used:
        tag = classify_usage(u, echo)
        counts[tag] = counts.get(tag, 0) + 1
    return _esc(", ".join(f"{n}×{tag}" for tag, n in counts.items()))


def _echoes_section(
    echoes_path: Path, repo: GraphRepo, usage_path: Path | None = None,
) -> str:
    """Render the recent bridge-invocations panel from ``echoes.jsonl`` (E.1
    data — one line per priming call). Most recent first. Resolves primed
    record ids to short summaries via the read-only repo. When ``usage_path``
    is given, ``usage.jsonl`` active-use entries are joined onto each turn and
    summarized in a ``used`` column. Absent file (or the ``PRIMING_STREAM_ECHOES_OFF``
    kill-switch, which simply stops the file growing) → a graceful empty note."""
    echoes = _read_echoes(echoes_path, _ECHOES_LIMIT)
    if not echoes:
        return (
            "<h2>Recent bridge invocations</h2>"
            '<p class="empty">No bridge invocations recorded yet.</p>'
        )
    if usage_path is not None:
        echoes, _orphans = attach_usage_to_echoes(echoes, read_usage(usage_path))
    rows = []
    for e in reversed(echoes):  # most recent first
        sess = _esc((e.get("session_id") or "")[:8] or "?")
        ms = e.get("spread_ms")
        ms_s = f"{ms:.0f}ms" if isinstance(ms, (int, float)) else "&mdash;"
        n_a = len(e.get("semantic") or [])
        n_b = len(e.get("lexical") or [])
        # Resolve up to a few primed ids to summaries for the at-a-glance row.
        primed_ids = list(e.get("semantic") or []) + list(e.get("lexical") or [])
        previews = []
        for rid in primed_ids[:3]:
            rec = repo.get_record(rid)
            if rec is not None and rec.summary:
                previews.append(_truncate(rec.summary.replace("\n", " "), 80))
        preview = _esc("; ".join(previews)) if previews else "&mdash;"
        rows.append(
            f"<tr><td>{_esc(e.get('at', '?'))}</td>"
            f"<td>{sess}</td>"
            f"<td>{_esc(e.get('source', '?'))}</td>"
            f"<td>{_esc(ms_s)}</td>"
            f"<td>{_esc(n_a)}/{_esc(n_b)}</td>"
            f"<td>{_used_cell(e)}</td>"
            f'<td class="summary">{_esc(_truncate(e.get("prompt_head", ""), 80))}</td>'
            f'<td class="summary">{preview}</td></tr>'
        )
    body = "".join(rows)
    return (
        "<h2>Recent bridge invocations</h2>"
        f"<p>Showing latest {len(echoes)} primings (most recent first); "
        "A/B = semantic/lexical record counts; used = active substrate reads "
        "(usage.jsonl) joined to the turn.</p>"
        "<table><tr>"
        "<th>at</th><th>session</th><th>source</th><th>spread</th>"
        "<th>A/B</th><th>used</th><th>prompt head</th><th>primed (sample)</th>"
        f"</tr>{body}</table>"
    )


def generate_dashboard(
    db_path: Path,
    out: Path,
    vec_index: RecordsVecIndex | None = None,
) -> Path:
    """Render the v0.7-x-vec-index active graph to a static HTML file.

    ``db_path`` points at the SQLite database (records + sleep_cycles).
    ``out`` is the HTML file to write; parent directories are created.
    ``vec_index`` is an injection seam for tests; when ``None`` a default
    :class:`RecordsVecIndex` is constructed (so the existing ``Priming Stream
    dashboard`` CLI keeps working without arg changes). If construction
    fails the panel renders ``unavailable`` rather than crashing.
    """
    db_path = Path(db_path)
    out = Path(out)
    cfg = load_config()
    paths = resolve_paths(cfg)

    if vec_index is None:
        try:
            vec_index = RecordsVecIndex(
                paths.vec_index_dir, cfg.vec_index.model_name,
            )
        except Exception:  # noqa: BLE001 - degrade
            vec_index = None

    conn = _connect_readonly(db_path)
    try:
        repo = GraphRepo(conn)
        echoes_path = paths.episodic_dir / "echoes.jsonl"
        usage_path = paths.episodic_dir / "usage.jsonl"
        sections = [
            _health_section(conn, vec_index, paths.vec_index_dir),
            _records_table(repo),
            _cycles_table(repo),
            _echoes_section(echoes_path, repo, usage_path),
        ]
    finally:
        conn.close()

    document = (
        '<!DOCTYPE html>\n<html lang="en"><head>'
        '<meta charset="utf-8">'
        "<title>Priming Stream Inspector</title>"
        f"<style>{_CSS}</style></head><body>"
        "<h1>Priming Stream &mdash; Inspector</h1>"
        f"<p>Graph database: {_esc(db_path)}</p>"
        + "".join(sections)
        + "</body></html>\n"
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document, encoding="utf-8")
    return out
