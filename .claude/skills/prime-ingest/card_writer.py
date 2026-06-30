"""Bulk card-writer (piece3 Phase C). The doc analog of writer.py.

Each doc-ingest worker writes ONE results file
``storage/corpus/_doc_results/<cardstem>.txt`` containing the index-card
BODY as plain markdown (## Summary / ## Key points) — NOT JSON, so quotes /
newlines never break parsing. This step pairs each body with its assignment
(doc_key, source, content_hash, created_at) and STAGES the card row
(``records_staging`` — SQL-canonical, 2026-06-12). ``stage_record`` drops
any prior staged card with the same doc_key first, so a regen/re-run
replaces in place (the Phase-A one-card-per-doc_key invariant).

``sleep-finalize`` then promotes staging → SQLite ``records`` + ChromaDB
(upsert by doc_key) — no Priming Stream change.

Over-budget cards (body > ~80 words) are LOGGED, not truncated — visibility
for calibration, no silent cap.

Usage:  python card_writer.py
"""
from __future__ import annotations

import json
from pathlib import Path

from priming_stream.core.config import load_config
from priming_stream.core.db import connect
from priming_stream.core.graph_repo import GraphRepo
from priming_stream.core.models import Record, new_record_id
from priming_stream.core.paths import resolve_paths
from priming_stream.core.schema import apply_migrations
from priming_stream.ingest.doc_ingest import canonical_doc_key, card_md_filename

WORD_BUDGET = 100  # hard ceiling per the document-mode contract


def _parse_card_result(text: str):
    """Split a C-worker result into identity COMPONENTS + the card body.

    Format: a header of ``DOI:/URL:/AUTHORS:/YEAR:/DOCTITLE:`` lines, then a
    ``===CARD===`` delimiter, then the card body. Returns ``(components, body)``
    or ``(None, None)`` if the delimiter is absent (the worker must emit it so
    the canonical doc_key can be derived in Python — single source of truth,
    matching the conversational flow)."""
    if "===CARD===" not in text:
        return None, None
    head, _, body = text.partition("===CARD===")
    comp = {"doi": None, "url": None, "authors": None, "year": None, "title": None}
    keymap = {"DOI:": "doi", "URL:": "url", "AUTHORS:": "authors",
              "YEAR:": "year", "DOCTITLE:": "title"}
    for ln in head.splitlines():
        s = ln.strip()
        for prefix, fld in keymap.items():
            if s.startswith(prefix):
                comp[fld] = s[len(prefix):].strip() or None
                break
    return comp, body.strip()


def main() -> None:
    cfg = load_config()
    paths = resolve_paths(cfg)
    corpus = Path(paths.graph_db).parent / "corpus"
    index_path = corpus / "_doc_index.json"

    if not index_path.exists():
        print("no _doc_index.json — run doc_plan.py first")
        return

    conn = connect(paths.graph_db)
    try:
        apply_migrations(conn)
        repo = GraphRepo(conn)
        _run(repo, index_path, json)
    finally:
        conn.close()


def _run(repo, index_path, json) -> None:
    index = json.loads(index_path.read_text(encoding="utf-8"))["docs"]
    written = 0
    missing = empty = suspect = 0
    over_budget: list[tuple[str, int]] = []
    over_bullets: list[tuple[str, int]] = []
    suspect_files: list[str] = []
    BULLET_CAP = 5

    for entry in index:
        assign_path = Path(entry["assign_path"])
        if not assign_path.exists():
            missing += 1
            continue
        a = json.loads(assign_path.read_text(encoding="utf-8"))
        results_path = Path(a["results_path"])
        if not results_path.exists():
            missing += 1
            continue
        raw = results_path.read_text(encoding="utf-8").strip()
        if not raw:
            empty += 1
            continue

        # piece3-C (canonical rewire): the worker emits identity COMPONENTS +
        # ===CARD=== + body. Derive the canonical doc_key in Python (single
        # source of truth, shared with the conversational flow) — never a key
        # the LLM slugged itself. fallback = the original file's stem so a key
        # always forms even with no metadata.
        comp, body = _parse_card_result(raw)
        if comp is None:
            suspect += 1
            suspect_files.append(Path(a["source"]).name)
            continue
        fallback = Path(a["source"].replace("file:///", "")).stem
        try:
            doc_key = canonical_doc_key(
                doi=comp["doi"], url=comp["url"], authors=comp["authors"],
                year=comp["year"], title=comp["title"], fallback=fallback,
            )
        except ValueError:
            suspect += 1
            suspect_files.append(fallback)
            continue
        card_filename = card_md_filename(doc_key)

        # No-preamble guard: the body must start with a markdown header.
        if not body or not body.lstrip().startswith("#"):
            suspect += 1
            suspect_files.append(card_filename)
            continue

        words = len(body.split())
        if words > WORD_BUDGET:
            over_budget.append((card_filename, words))
        n_bullets = sum(
            1 for ln in body.splitlines() if ln.lstrip().startswith("- ")
        )
        if n_bullets > BULLET_CAP:
            over_bullets.append((card_filename, n_bullets))

        repo.stage_record(Record(
            id=new_record_id(),
            source_uri=a["source_uri"],
            anchor_offset_start=0,      # doc-level node, not a chunk span
            anchor_offset_end=0,
            summary=body,
            created_at=a["created_at"],
            kind="index_card",
            doc_key=doc_key,
            source=a["source"] or None,
            content_hash=a["content_hash"] or None,
            title=comp["title"],
            provisional=False,
        ))
        written += 1

    print(f"card-write: {written} cards staged")
    if missing or empty or suspect:
        print(f"  skipped: missing_results={missing} empty_results={empty} "
              f"suspect_body={suspect}")
    if suspect_files:
        print(f"  suspect bodies (no leading '#', not written — re-run these):")
        for fn in suspect_files[:20]:
            print(f"    {fn}")
    if over_bullets:
        print(f"  over-bullets (>{BULLET_CAP}, the BINDING cap, kept not truncated): {len(over_bullets)}")
        for fn, n in sorted(over_bullets, key=lambda kv: -kv[1])[:20]:
            print(f"    {n:>2d} bullets  {fn}")
    if over_budget:
        print(f"  (informational) >{WORD_BUDGET}w on disk incl markdown: {len(over_budget)}")
        for fn, w in sorted(over_budget, key=lambda kv: -kv[1])[:20]:
            print(f"    {w:>4d}w  {fn}")


if __name__ == "__main__":
    main()
