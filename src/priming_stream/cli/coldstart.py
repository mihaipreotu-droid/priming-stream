"""``prime coldstart`` — deterministic plumbing for v0.7-x-vec-index.

No LLM. No per-chunk driver loop. The flow is:

1. Parse a TOML manifest pointing at one or more claude.ai exports.
2. For each export, run :class:`ClaudeAiExportAdapter` and append every
   chunk to the episodic store's ``chunks.jsonl`` (idempotent on
   ``chunk_id``).
3. Drain pending chunks to ``.md`` files under ``storage/corpus/imports``
   via :func:`materialize_pending`.
4. Print a summary.

Indexing is **out of scope here** — v0.7-x-vec-index only indexes records
into ChromaDB, and records come from the ``/prime-ingest`` skill (LLM
extraction), not from coldstart. Chunks live on disk and are reached via
``graph_chunk_around_anchor`` tier-2.

Record extraction is **out of scope here** — it runs separately as the
``/prime-ingest`` skill inside an active Claude Code session.
"""
from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

from priming_stream.core.config import load_config
from priming_stream.core.episodic import EpisodicStore
from priming_stream.core.paths import ensure_dirs, resolve_paths
from priming_stream.ingest.claude_ai_export import ClaudeAiExportAdapter
from priming_stream.ingest.materialize import materialize_pending


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "coldstart",
        help="materialize claude.ai exports into storage/corpus/imports",
    )
    p.add_argument(
        "--config", required=True,
        help="path to a coldstart TOML manifest",
    )
    p.set_defaults(func=handle_coldstart)


# -- manifest ------------------------------------------------------------


def _load_manifest(path: Path) -> dict:
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    exports = raw.get("exports", {})
    paths_raw = exports.get("paths", []) if isinstance(exports, dict) else []
    out_paths: list[str] = []
    if isinstance(paths_raw, list):
        for p in paths_raw:
            if isinstance(p, str) and p:
                out_paths.append(Path(p).expanduser().as_posix())
    return {"exports": out_paths, "raw": raw}


# -- steps ---------------------------------------------------------------


def _ingest_exports(
    export_paths: list[str], store: EpisodicStore, cfg,
) -> int:
    """Walk every export, write each chunk into ``chunks.jsonl``.

    ``EpisodicStore.write_chunk`` is idempotent on ``chunk_id``, so
    re-running coldstart on the same export does not duplicate entries.
    Returns the number of chunks visited (pre-dedupe).
    """
    seen = 0
    for export in export_paths:
        export_path = Path(export)
        if not export_path.exists():
            print(
                f"[coldstart] warn: export missing — {export}",
                file=sys.stderr,
            )
            continue
        adapter = ClaudeAiExportAdapter(
            export_path,
            idle_minutes=cfg.sleep.idle_minutes,
            chunk_max_turns=cfg.sleep.chunk_max_turns,
            chunk_max_chars=cfg.sleep.chunk_max_chars,
        )
        for chunk in adapter.iter_chunks():
            store.write_chunk(chunk)
            seen += 1
        print(f"[coldstart] ingested export: {export}")
    return seen


def _materialize_chunks(
    store: EpisodicStore, imports_root: Path, cursor_path: Path,
) -> list[Path]:
    return materialize_pending(store, imports_root, cursor_path)


def _finalize_imports(materialized: list[Path]) -> None:
    """No-op finalizer (v0.7-x-vec-index drops the qmd indexing step).

    Kept as a named step so coldstart's flow reads ``ingest →
    materialize → finalize`` and a future post-materialize hook has an
    obvious home.
    """
    print(f"[coldstart] finalized {len(materialized)} materialized chunks")


# -- main handler --------------------------------------------------------


def handle_coldstart(args: argparse.Namespace) -> int:
    try:
        cfg = load_config()
        paths = resolve_paths(cfg, project_root=Path.cwd())
        if not paths.graph_db.exists():
            print(
                f"no graph database at {paths.graph_db} — "
                f"run 'prime init' first",
                file=sys.stderr,
            )
            return 1
        ensure_dirs(paths)

        cfg_path = Path(args.config).expanduser().resolve()
        if not cfg_path.is_file():
            print(
                f"[coldstart] config not found: {cfg_path}",
                file=sys.stderr,
            )
            return 1
        manifest = _load_manifest(cfg_path)
        export_paths = manifest["exports"]
        if not export_paths:
            print(
                "[coldstart] manifest has no [exports] paths",
                file=sys.stderr,
            )
            return 1

        store = EpisodicStore(paths.episodic_dir)

        ingested = _ingest_exports(export_paths, store, cfg)
        print(f"[coldstart] {ingested} chunks attempted (idempotent on id)")

        materialized = _materialize_chunks(
            store, paths.corpus_imports_dir, paths.corpus_cursor_path,
        )
        print(f"[coldstart] {len(materialized)} chunks materialized to .md")

        _finalize_imports(materialized)
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"coldstart failed: {exc}", file=sys.stderr)
        return 1
