"""Document-ingest planner (piece3 Phase C). The doc analog of prime-sleep/plan.py.

Turns a directory of ORIGINAL documents into per-document index-card
extraction assignments for the doc-ingest Workflow.

Identity model (locked 2026-06-01 — "do it right"):
  - The ORIGINAL document (PDF/docx/… or a native .md note) is the identity,
    the source link, AND what content_hash is taken over. doc_key =
    ``path:<abs original>``; source / source_uri = the original.
  - The ``.md`` the worker reads is an EPHEMERAL working translation — either
    an existing conversion (mapped from a parallel conversions dir) or one
    markitdown generates on the fly. Its path is NOT stored on the card.

Per original:
  - doc_key = derive_doc_key(original); content_hash = sha256(original bytes).
  - prefilter against existing cards (skip carded + unchanged → no LLM call).
  - resolve the worker-input .md (native .md → itself; else conversion or
    markitdown); route Sonnet/Opus by the .md's body-token load.
  - write one per-doc assignment file + a doc-index the Workflow reads.

Usage:  python doc_plan.py --originals <dir> [--conversions <dir>]
                           [--threshold 100000] [--ext .pdf,.docx,...]
Writes: storage/corpus/_doc_assign/<cardstem>.json
        storage/corpus/_doc_index.json
"""
from __future__ import annotations

import argparse
import shutil
import sys
import warnings
from pathlib import Path

from priming_stream.core.config import load_config
from priming_stream.core.db import connect
from priming_stream.core.graph_repo import GraphRepo
from priming_stream.core.models import now_iso
from priming_stream.core.paths import resolve_paths
from priming_stream.core.source_uri import build as build_uri
from priming_stream.ingest.doc_ingest import content_hash

CHARS_PER_TOK = 3.8
TOKEN_THRESHOLD = 100_000  # .md body tokens; > this -> Opus, else Sonnet
# Original document types we ingest. A native .md/.txt is its own worker
# input (no conversion step). Everything else needs a .md translation.
ORIGINAL_EXTS = {
    ".pdf", ".docx", ".doc", ".odt", ".pptx", ".ppt", ".xlsx", ".csv",
    ".html", ".htm", ".md", ".txt", ".rtf", ".epub",
}
NATIVE_MD_EXTS = {".md", ".txt"}


def _file_uri(p: Path) -> str:
    return build_uri("file", path=p.resolve().as_posix())


def _find_conversion(
    original: Path, originals_root: Path, conversions_root: Path | None,
) -> Path | None:
    """Map an original to an existing .md conversion by parallel relpath+stem
    (``<orig_root>/A/x.pdf`` -> ``<conv_root>/A/x.md``). None if absent."""
    if conversions_root is None:
        return None
    try:
        rel = original.resolve().relative_to(originals_root.resolve())
    except ValueError:
        return None
    cand = (conversions_root / rel).with_suffix(".md")
    return cand if cand.exists() else None


def _markitdown_to(original: Path, out_dir: Path) -> Path | None:
    """Convert an original to .md via markitdown into the ephemeral work dir.
    Returns the .md path, or None on failure (logged; the doc is skipped)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (original.stem + ".md")
    try:
        warnings.filterwarnings("ignore")
        from markitdown import MarkItDown
        text = MarkItDown().convert(str(original)).text_content
        out.write_text(text or "", encoding="utf-8")
        return out
    except Exception as exc:  # noqa: BLE001 - per-doc boundary
        print(f"  WARN markitdown failed for {original.name}: {exc}", file=sys.stderr)
        return None


def _text_len(md_path: Path) -> int:
    # Byte length is a fine proxy for the token estimate (routing only) and
    # is robust to non-utf-8 conversions (some markitdown output isn't clean
    # utf-8). The worker reads the file via the Read tool, which handles
    # encoding itself — doc_plan only needs a size for Sonnet/Opus routing.
    try:
        return len(md_path.read_bytes())
    except OSError:
        return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--originals", action="append", default=None,
        help="dir of original docs (recursive), or a single file. Repeatable: "
             "pass --originals multiple times to ingest several scattered "
             "folders/files in one cycle (each path is a file OR a recursed dir).",
    )
    ap.add_argument(
        "--originals-list", default=None,
        help="path to a JSON array of individual original-file paths (scattered "
             "files, e.g. the conversation branch's _produced_docs.json); merged "
             "with --originals, deduped, filtered to the doc-type allowlist",
    )
    ap.add_argument("--conversions", default=None, help="dir of existing .md conversions (parallel layout)")
    ap.add_argument("--threshold", type=int, default=TOKEN_THRESHOLD)
    ap.add_argument("--ext", default=None, help="comma-separated override of original extensions")
    ap.add_argument(
        "--no-generate", action="store_true",
        help="do NOT markitdown-convert originals lacking a mapped .md; "
             "skip + report them instead (fast dry-run / use existing only)",
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="cap the number of docs planned (after prefilter) — for a "
             "small controlled dry-run before a full corpus ingest",
    )
    args = ap.parse_args()

    conversions_root = Path(args.conversions).resolve() if args.conversions else None
    exts = (
        {e if e.startswith(".") else "." + e for e in args.ext.split(",")}
        if args.ext else ORIGINAL_EXTS
    )
    if not args.originals and not args.originals_list:
        ap.error("provide --originals and/or --originals-list")

    cfg = load_config()
    paths = resolve_paths(cfg)
    corpus = Path(paths.graph_db).parent / "corpus"
    contract_path = str(Path(__file__).resolve().parents[3] / "prompts" / "extract_record.md")

    assign_dir = corpus / "_doc_assign"
    results_dir = corpus / "_doc_results"
    md_workdir = corpus / "_doc_md"
    for d in (assign_dir, results_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    md_workdir.mkdir(parents=True, exist_ok=True)

    # Read-only repo connection for the prefilter (existing-card check).
    conn = connect(paths.graph_db)
    repo = GraphRepo(conn)

    created_at = now_iso()
    # Build the originals list from --originals (a single file OR a recursed
    # directory) and/or --originals-list (a JSON array of scattered file paths,
    # e.g. the conversation branch's _produced_docs.json). ``rel_root`` is what
    # conversion relpaths are computed against (moot for list files / no convs).
    import json as _json
    originals: list[Path] = []
    # Per-original conversion root: a file maps against its parent dir, a folder
    # against itself. With several --originals roots, each original remembers the
    # root it came from so _find_conversion uses the right parallel layout. On the
    # pathological overlap (same file given both directly AND inside a passed
    # folder) the FIRST registration wins the rel_root — harmless: dedupe keeps one
    # copy, and conversion-relpath only matters for non-.md originals + --conversions.
    default_rel_root: Path = Path(".").resolve()
    rel_root_for: dict[Path, Path] = {}
    seen: set[Path] = set()
    for raw_root in args.originals or []:
        originals_root = Path(raw_root).resolve()
        if originals_root.is_file():
            if originals_root.suffix.lower() in exts and originals_root not in seen:
                originals.append(originals_root)
                seen.add(originals_root)
                rel_root_for[originals_root] = originals_root.parent
        else:
            for p in sorted(
                q for q in originals_root.rglob("*")
                if q.is_file() and q.suffix.lower() in exts
            ):
                rp = p.resolve()
                if rp in seen:
                    continue
                originals.append(rp)
                seen.add(rp)
                rel_root_for[rp] = originals_root
    if args.originals_list:
        try:
            listed = _json.loads(
                Path(args.originals_list).read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            listed = []
        for raw in listed if isinstance(listed, list) else []:
            if not isinstance(raw, str):
                continue
            p = Path(raw).expanduser()
            if (p.is_file() and p.suffix.lower() in exts
                    and p.resolve() not in seen):
                originals.append(p.resolve())
                seen.add(p.resolve())

    import hashlib
    import json
    index = []
    n_skip = n_gen = n_convfound = n_native = n_noconv = 0
    for original in originals:
        try:
            chash = content_hash(original.read_bytes())
        except OSError:
            continue
        # piece3-C (canonical rewire): prefilter on content_hash — the canonical
        # doc_key is derived LATER from metadata the worker extracts, so it's
        # unknown here. An unchanged, already-carded doc hashes the same -> skip.
        if repo.card_exists_with_content_hash(chash):
            n_skip += 1
            continue

        ext = original.suffix.lower()
        if ext in NATIVE_MD_EXTS:
            md_path = original
            n_native += 1
        else:
            md_path = _find_conversion(
                original, rel_root_for.get(original, default_rel_root),
                conversions_root,
            )
            if md_path is not None:
                n_convfound += 1
            elif args.no_generate:
                n_noconv += 1
                continue
            else:
                md_path = _markitdown_to(original, md_workdir)
                if md_path is None:
                    n_noconv += 1
                    continue
                n_gen += 1

        est_tokens = int(_text_len(md_path) / CHARS_PER_TOK)
        mode = "opus" if est_tokens > args.threshold else "sonnet"
        src_uri = _file_uri(original)
        # Deterministic stem for the assign/results pair (doc_key is unknown
        # until card_writer derives it from the worker's components).
        stem = "doc_" + hashlib.sha256(src_uri.encode()).hexdigest()[:12]

        slice_obj = {
            "source": src_uri,             # original — the canonical link
            "source_uri": src_uri,         # card frontmatter source_uri
            "content_hash": chash,
            "md_path": str(md_path),       # EPHEMERAL worker input
            "mode": mode,
            "est_tokens": est_tokens,
            "results_path": str(results_dir / f"{stem}.txt"),
            "contract_path": contract_path,
            "created_at": created_at,
        }
        assign_path = assign_dir / f"{stem}.json"
        assign_path.write_text(json.dumps(slice_obj, ensure_ascii=False), encoding="utf-8")
        index.append({
            "assign_path": str(assign_path), "mode": mode,
            "source": src_uri, "est_tokens": est_tokens,
        })
        if args.limit is not None and len(index) >= args.limit:
            break

    conn.close()

    index_path = corpus / "_doc_index.json"
    index_path.write_text(json.dumps({"docs": index}, ensure_ascii=False), encoding="utf-8")

    print(f"index={index_path}")
    print(f"originals_found={len(originals)}  to_card={len(index)}  "
          f"skipped_unchanged={n_skip}")
    print(f"  conversion: existing={n_convfound} generated={n_gen} "
          f"native_md={n_native} no_conversion_skipped={n_noconv}")
    op = sum(1 for e in index if e["mode"] == "opus")
    print(f"  routing: sonnet={len(index)-op} opus={op}")
    for e in index[:60]:
        # doc_key is unknown at plan time (card_writer derives it from the
        # worker's metadata, post-rewire); show the source path instead.
        print(f"  {e['mode']:6s} ~tok={e['est_tokens']:>6d}  {e['source']}")


if __name__ == "__main__":
    main()
