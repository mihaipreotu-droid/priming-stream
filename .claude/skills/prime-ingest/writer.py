"""Bulk-writer — materializes the workers' plain-text results into STAGED
records (the ``records_staging`` table — SQL-canonical, 2026-06-12).

Each extraction worker writes ONE file storage/corpus/_sleep_results/<conv>.txt
in a delimited PLAIN-TEXT format (NOT JSON — so quotes/commas/newlines in
summaries can never break parsing). Format:

    CONV: <conv>
    NOTABLE: yes|no
    NOTE: <one line>
    ===REC===
    CHUNK: <chunk_id>
    ANCHOR: <start> <end>
    <summary, free text>
    ===REC===
    ...

This step pairs each record with the conversation's pre-generated rec_id +
source_uri + created_at (from the assignment), clamps anchors, and stages the
rows in bulk (INSERT OR REPLACE — a re-run over the same assignments is
idempotent, the staging analog of overwriting the same ``.md`` file).
``sleep-finalize`` then promotes staging → ``records`` — no Priming Stream change.

Usage:  python writer.py
"""
from __future__ import annotations

from pathlib import Path

from priming_stream.core.config import load_config
from priming_stream.core.paths import resolve_paths


def _body_len(path: str) -> int:
    try:
        txt = Path(path).read_text(encoding="utf-8")
    except OSError:
        return 0
    if txt.startswith("---"):
        end = txt.find("\n---", 3)
        if end != -1:
            nl = txt.find("\n", end + 1)
            return len(txt[nl + 1:]) if nl != -1 else 0
    return len(txt)


def _parse_block(block: str):
    """Parse one ===REC=== block.

    Returns ``(chunk_id, start, end, doc_ref, summary)`` or None. piece3-B: an
    optional ``DOCREF:`` line (the TITLE of the document this record is built
    on) may sit between the ANCHOR line and the summary. The canonical doc_key
    is derived in ``main`` from the matching ===DOC=== block's components — the
    worker never emits the key itself (LLM slugs are non-deterministic and would
    diverge from the Python ``canonical_doc_key`` used by the doc-ingest flow).
    """
    lines = block.strip("\n").split("\n")
    chunk_id = None
    start = end = 0
    anchor_idx = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if chunk_id is None and s.startswith("CHUNK:"):
            chunk_id = s[len("CHUNK:"):].strip()
        elif chunk_id is not None and s.startswith("ANCHOR:"):
            parts = s[len("ANCHOR:"):].split()
            try:
                start = int(parts[0]); end = int(parts[1]) if len(parts) > 1 else start
            except (ValueError, IndexError):
                start = end = 0
            anchor_idx = i
            break
    if chunk_id is None or anchor_idx is None:
        return None
    doc_ref = None
    j = anchor_idx + 1
    while j < len(lines):
        s = lines[j].strip()
        if s.startswith("DOCREF:"):
            doc_ref = s[len("DOCREF:"):].strip() or None; j += 1
        else:
            break
    summary = "\n".join(lines[j:]).strip()
    if not summary:
        return None
    return chunk_id, start, end, doc_ref, summary


def _classify_blocks(text: str):
    """Split on BOTH delimiters and route each block by its own delimiter, so a
    ===DOC=== interleaved among records (worker misordering) never swallows the
    records that follow it. Returns ``(rec_blocks, doc_blocks)``; the leading
    header (before the first delimiter) is discarded."""
    import re
    parts = re.split(r"(===REC===|===DOC===)", text)
    recs: list[str] = []
    docs: list[str] = []
    i = 1
    while i < len(parts):
        delim = parts[i]
        block = parts[i + 1] if i + 1 < len(parts) else ""
        (recs if delim == "===REC===" else docs).append(block)
        i += 2
    return recs, docs


def _parse_doc_block(block: str):
    """Parse one ===DOC=== block -> a dict of identity COMPONENTS + stub body.

    The worker emits components (DOI/URL/AUTHORS/YEAR/DOCTITLE/SOURCE), NOT the
    final key — ``main`` derives the canonical key via ``canonical_doc_key`` so
    the derivation is a single source of truth shared with the document-ingest
    (C) flow. A non-empty body after the header → a stub CARD; an empty body →
    a tag-only document (it still provides the key for record DOCREFs, no card).
    Returns ``{doi,url,authors,year,title,source,stub}`` or None (title required).
    """
    lines = block.strip("\n").split("\n")
    fields: dict = {"doi": None, "url": None, "authors": None,
                    "year": None, "title": None, "source": None,
                    "local_path": None}
    keymap = {"DOI:": "doi", "URL:": "url", "AUTHORS:": "authors",
              "YEAR:": "year", "DOCTITLE:": "title", "SOURCE:": "source",
              "LOCALPATH:": "local_path"}
    j = 0
    while j < len(lines):
        s = lines[j].strip()
        matched = False
        for prefix, fld in keymap.items():
            if s.startswith(prefix):
                fields[fld] = s[len(prefix):].strip() or None
                matched = True
                break
        if not matched:
            break
        j += 1
    if not fields["title"]:
        return None
    fields["stub"] = "\n".join(lines[j:]).strip()  # "" => tag-only
    return fields


def _chunk_cwd(chunk_path: str) -> str | None:
    """Read the CC working directory from a materialized chunk's frontmatter
    (``cwd:`` line), or None. The carrier that lets a produced-doc basename —
    absent from the conversation text — resolve to a full path."""
    try:
        txt = Path(chunk_path).read_text(encoding="utf-8")
    except OSError:
        return None
    if not txt.startswith("---"):
        return None
    end = txt.find("\n---", 3)
    if end == -1:
        return None
    for ln in txt[3:end].splitlines():
        k, sep, v = ln.partition(":")
        if sep and k.strip() == "cwd":
            return v.strip() or None
    return None


def _chunk_doc_paths(chunk_path: str) -> list[str]:
    """Read ``doc_paths`` (a JSON array) from a chunk's frontmatter, or [].
    These are full paths of document-type files the session touched via Read/
    Write/Edit tools — for resolving an input/output doc's basename to a real
    path even when it isn't in cwd."""
    try:
        txt = Path(chunk_path).read_text(encoding="utf-8")
    except OSError:
        return []
    if not txt.startswith("---"):
        return []
    end = txt.find("\n---", 3)
    if end == -1:
        return []
    import json as _json
    for ln in txt[3:end].splitlines():
        k, sep, v = ln.partition(":")
        if sep and k.strip() == "doc_paths":
            try:
                val = _json.loads(v.strip())
            except ValueError:
                return []
            return [str(x) for x in val] if isinstance(val, list) else []
    return []


def _find_in_tree(root: Path, basename: str) -> str | None:
    """First file named ``basename`` anywhere under ``root`` — a bounded subdir
    search (skips heavy dirs) for a skill-produced doc not in the cwd root."""
    import os
    skip = {".git", "node_modules", "__pycache__", ".venv", "venv",
            ".mypy_cache", "storage", "vec_index", "dist", "build"}
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d for d in dirnames if d not in skip and not d.startswith(".")
            ]
            if basename in filenames:
                return str((Path(dirpath) / basename).resolve())
    except OSError:
        return None
    return None


def _resolve_doc_path(
    local_path: str | None, cwd: str | None,
    doc_paths: list[str] | None = None,
) -> str | None:
    """Resolve a doc-candidate's LOCALPATH (usually just a basename) to an
    existing file, or None. Order:
      1. an existing absolute path as-is;
      2. a tool-event full path whose basename matches — covers INPUT docs read
         from anywhere + OUTPUT docs written elsewhere (Read/Write/Edit paths the
         adapter surfaced);
      3. cwd + value / cwd + basename — skill-produced docs in the working dir;
      4. a bounded subdir walk of cwd — skill-produced docs in a subdir.
    None when nothing resolves — the filesystem is the ground truth."""
    if not local_path:
        return None
    p = Path(local_path).expanduser()
    if p.is_file():
        return str(p.resolve())
    base = Path(local_path).name
    for tp in (doc_paths or []):
        if Path(tp).name == base:
            tpp = Path(tp).expanduser()
            if tpp.is_file():
                return str(tpp.resolve())
    if cwd:
        # LOCALPATH is LLM-emitted: confine the join to cwd (no ../ traversal).
        cwd_resolved = Path(cwd).resolve()
        for cand in (Path(cwd) / local_path, Path(cwd) / base):
            resolved = cand.resolve()
            if not resolved.is_relative_to(cwd_resolved):
                continue
            if resolved.is_file():
                return str(resolved)
        found = _find_in_tree(Path(cwd), base)
        if found:
            return found
    return None


def main() -> None:
    cfg = load_config()
    paths = resolve_paths(cfg)
    corpus = Path(paths.graph_db).parent / "corpus"
    results_dir = corpus / "_sleep_results"
    assign_dir = corpus / "_sleep_assign"

    import json
    from priming_stream.core.db import connect
    from priming_stream.core.graph_repo import GraphRepo
    from priming_stream.core.models import Record, new_record_id
    from priming_stream.core.schema import apply_migrations
    from priming_stream.ingest.doc_ingest import canonical_doc_key

    conn = connect(paths.graph_db)
    try:
        apply_migrations(conn)
        repo = GraphRepo(conn)
        _run(repo, corpus, results_dir, assign_dir, json,
             new_record_id, canonical_doc_key, Record)
    finally:
        conn.close()


def _run(repo, corpus, results_dir, assign_dir, json,
         new_record_id, canonical_doc_key, Record) -> None:
    total = 0
    over_pool = bad_chunk = bad_block = bad_doc = 0
    per_conv: dict[str, int] = {}
    # piece3-B: enforce the principle "card a doc ONLY when a record is built
    # on it" GLOBALLY. Accumulate every doc-candidate (key -> fields) and the
    # set of keys actually referenced by some record (DOCREF); after the loop,
    # write a stub card only for a doc that BOTH has a stub body AND is
    # record-referenced. Orphan stubs (characterized but no record built on
    # them — passing web refs, untagged works) are dropped.
    referenced_keys: set[str] = set()
    all_docs: dict[str, tuple] = {}  # key -> (title, source, stub, created_at)

    for rf in sorted(results_dir.glob("*.txt")):
        conv = rf.stem
        text = rf.read_text(encoding="utf-8")
        # piece3-B: classify blocks by their own delimiter (order-independent),
        # so a misordered ===DOC=== never swallows trailing records.
        blocks, doc_blocks = _classify_blocks(text)
        assign_path = assign_dir / f"{conv}.json"
        if not assign_path.exists():
            print(f"  WARN no assignment for {conv}; skipping")
            continue
        a = json.loads(assign_path.read_text(encoding="utf-8"))
        rec_ids = a["rec_ids"]
        created_at = a["created_at"]
        uri_by_chunk = {c["chunk_id"]: c["source_uri"] for c in a["chunks"]}
        path_by_chunk = {c["chunk_id"]: c["path"] for c in a["chunks"]}
        # session working dir (same across a conversation's chunks) — for
        # resolving produced-doc basenames to full paths.
        conv_cwd = next(
            (c for c in (_chunk_cwd(p) for p in path_by_chunk.values()) if c),
            None,
        )
        conv_docs = sorted({
            d for p in path_by_chunk.values() for d in _chunk_doc_paths(p)
        })
        len_cache: dict[str, int] = {}

        # piece3-B: parse ===DOC=== blocks FIRST and derive the canonical key in
        # Python (single source of truth). Build a title -> (key, title) map so
        # records' DOCREF (a title) resolves to the canonical key.
        title_to_doc: dict[str, tuple[str, str]] = {}
        for block in doc_blocks:
            f = _parse_doc_block(block)
            if f is None:
                bad_doc += 1
                continue
            try:
                dk = canonical_doc_key(
                    doi=f["doi"], url=f["url"], authors=f["authors"],
                    year=f["year"], title=f["title"], fallback=f["title"],
                )
            except ValueError:
                bad_doc += 1
                continue
            title_to_doc[f["title"]] = (dk, f["title"])
            all_docs[dk] = (f["title"], f["source"], f["stub"], created_at,
                            f.get("local_path"), conv_cwd, conv_docs)

        n = 0
        for block in blocks:
            parsed = _parse_block(block)
            if parsed is None:
                bad_block += 1
                continue
            cid, start, end, doc_ref, summary = parsed
            uri = uri_by_chunk.get(cid)
            if uri is None:
                bad_chunk += 1
                continue
            if n >= len(rec_ids):
                over_pool += 1
                continue
            rid = rec_ids[n]
            if cid not in len_cache:
                len_cache[cid] = _body_len(path_by_chunk.get(cid, ""))
            blen = len_cache[cid] or 0
            start = max(0, start)
            if blen:
                end = min(end, blen)
            end = max(end, start)
            # piece3-B: a claim built on a doc carries the doc's canonical
            # doc_key + title. DOCREF is the title handle; resolve it to the
            # key derived from the matching ===DOC=== block. If unmatched
            # (worker named a doc it didn't declare), keep the title as a
            # weak ref, doc_key absent.
            claim_dk = claim_title = None
            if doc_ref:
                dk, dtitle = title_to_doc.get(doc_ref, (None, doc_ref))
                if dk:
                    claim_dk = dk
                    referenced_keys.add(dk)  # this doc IS built-on by a record
                claim_title = dtitle
            repo.stage_record(Record(
                id=rid,
                source_uri=uri,
                anchor_offset_start=start,
                anchor_offset_end=end,
                summary=summary,
                created_at=created_at,
                kind="claim",
                doc_key=claim_dk,
                title=claim_title,
            ))
            n += 1

        per_conv[conv] = n
        total += n

    # piece3-B: write stub cards AFTER all records, enforcing the principle —
    # a doc is carded only if (a) it has a stub body and (b) some record is
    # built on it (its key is in referenced_keys). Orphans are dropped.
    total_docs = orphans_dropped = tagonly = 0
    produced_paths: list[str] = []
    unresolved_docs: list[tuple[str, str]] = []
    for dk, (title, source, sbody, created_at, local_path, conv_cwd, conv_docs) in all_docs.items():
        referenced = dk in referenced_keys
        # produced/processed LOCAL document: a doc-candidate whose LOCALPATH
        # (often just a basename) resolves to a real file on disk → hand it to
        # the document branch to become a REAL index card (read from the file).
        # NOT gated on `referenced`: the worker already judged it final/
        # substantial and it exists on disk, so the produced artifact is notable
        # on its own — the fragile DOCREF↔DOCTITLE string match must not gate it
        # (a real-session run showed the worker's DOCREF not matching, which
        # would otherwise drop the produced deck). Resolution joins the session
        # cwd when only a basename is known; doc_plan's ORIGINAL_EXTS allowlist
        # drops any non-document path, so a stray code-file path never reaches
        # markitdown. (Independent of the stub body.)
        resolved = _resolve_doc_path(local_path, conv_cwd, conv_docs)
        if resolved:
            produced_paths.append(resolved)
        elif local_path:
            # the worker named a local doc but it didn't resolve to a real file
            # — surface it (silent drops hide missed produced/input docs)
            unresolved_docs.append((local_path, title or ""))
        # stub card (conversation-derived, NO local file) — keep the built-on
        # gate: an external reference earns a stub only if a record uses it.
        if not sbody:
            tagonly += 1
            continue  # tag-only document, no stub card by design
        if not referenced:
            orphans_dropped += 1
            continue  # characterized but no record built on it — drop the stub
        # stage_record drops any prior staged card with the same doc_key
        # first (the staging analog of the filename-keyed-by-doc_key
        # overwrite), so a re-run never duplicates a stub.
        repo.stage_record(Record(
            id=new_record_id(),
            source_uri="doc://" + dk,   # synthetic; stubs have no file
            anchor_offset_start=0,      # doc-level node, not a chunk span
            anchor_offset_end=0,
            summary=sbody,
            created_at=created_at,
            kind="index_card",
            doc_key=dk,
            source=source or None,
            content_hash=None,
            title=title,
            provisional=True,
        ))
        total_docs += 1

    # produced-doc handoff: scattered local files the conversation produced or
    # processed (final/substantial — the worker's judgment per the contract),
    # for the document branch to ingest as real cards THIS cycle. The reconcile
    # step then merges any stub ↔ real-card pair for the same document. Always
    # (re)write so an empty list clears a prior cycle's stale handoff.
    produced_unique = sorted(set(produced_paths))
    (corpus / "_produced_docs.json").write_text(
        json.dumps(produced_unique, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"bulk-write: {total} records + {total_docs} doc stubs staged")
    print(f"  docs: {len(all_docs)} candidates, {total_docs} carded, "
          f"{orphans_dropped} orphan-dropped, {tagonly} tag-only, "
          f"{len(produced_unique)} produced-local->doc-branch")
    for lp, title in unresolved_docs:
        print(f"  doc UNRESOLVED (named but not found on disk -> not carded): "
              f"'{lp}'" + (f" [{title[:40]}]" if title else ""))
    if over_pool or bad_chunk or bad_block or bad_doc:
        print(f"  skipped: over_pool={over_pool} bad_chunk_id={bad_chunk} "
              f"bad_block={bad_block} bad_doc={bad_doc}")
    for conv, n in sorted(per_conv.items(), key=lambda kv: -kv[1]):
        if n:
            print(f"  {conv[:13]:13s} {n}")


if __name__ == "__main__":
    main()
