"""``prime ingest-source`` — ingest a conversation source into the episodic
store (``chunks.jsonl``) WITHOUT materializing or touching the materialize
cursor.

This is the source-ingest leg of the unified ``ingest`` skill. It only ADDS
chunks to the episodic log; the ingest cycle's ``sleep-prepare`` is the single
place that materializes pending chunks to ``corpus/imports/`` and advances the
cursor. Keeping materialize OUT of here is deliberate: ``coldstart`` did both
(ingest + materialize + cursor-advance), which forced a manual ``_cursor.json``
reset before ``/prime-sleep`` could see anything (the W7 "critical glue"). One
materialize, in one place, removes that footgun.

Adapters by kind (both already exist; this just dispatches):
  --kind export  -> ClaudeAiExportAdapter (claude.ai conversation export)
  --kind cc      -> ClaudeCodeAdapter (Claude Code ``.jsonl`` session transcripts)

``--path`` is repeatable; each path is a folder (the adapter recurses) or a
single file. Ingest is idempotent on ``chunk_id`` (``EpisodicStore.write_chunk``),
so re-running over the same source is a no-op.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from priming_stream.core.config import load_config
from priming_stream.core.episodic import EpisodicStore
from priming_stream.core.paths import ensure_dirs, resolve_paths
from priming_stream.ingest.claude_ai_export import ClaudeAiExportAdapter
from priming_stream.ingest.claude_code import ClaudeCodeAdapter


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "ingest-source",
        help="ingest a conversation source (claude.ai export | Claude Code "
             "session) into the episodic store; does NOT materialize",
    )
    p.add_argument(
        "--kind", required=True, choices=["export", "cc"],
        help="source kind: 'export' = claude.ai conversation export; "
             "'cc' = Claude Code .jsonl session transcript(s)",
    )
    p.add_argument(
        "--path", required=True, action="append", dest="paths",
        metavar="PATH",
        help="source path (folder or file). Repeatable for multiple sources.",
    )
    p.set_defaults(func=cmd_ingest_source)


def _adapter_for(kind: str, path: Path, cfg):
    """Construct the source-appropriate adapter. ClaudeCodeAdapter takes no
    ``chunk_max_chars`` (its transcripts are turn-bounded); the export adapter
    needs it (a single claude.ai conversation can be one huge burst)."""
    if kind == "export":
        return ClaudeAiExportAdapter(
            path,
            idle_minutes=cfg.sleep.idle_minutes,
            chunk_max_turns=cfg.sleep.chunk_max_turns,
            chunk_max_chars=cfg.sleep.chunk_max_chars,
        )
    return ClaudeCodeAdapter(
        path,
        idle_minutes=cfg.sleep.idle_minutes,
        chunk_max_turns=cfg.sleep.chunk_max_turns,
    )


def cmd_ingest_source(args: argparse.Namespace) -> int:
    try:
        cfg = load_config()
        paths = resolve_paths(cfg, project_root=Path.cwd())
        if not paths.graph_db.exists():
            print(
                f"no graph database at {paths.graph_db} — run 'prime init' first",
                file=sys.stderr,
            )
            return 1
        ensure_dirs(paths)

        store = EpisodicStore(paths.episodic_dir)
        total = 0
        for raw in args.paths:
            src = Path(raw).expanduser()
            if not src.exists():
                print(
                    f"[ingest-source] warn: path missing — {src}",
                    file=sys.stderr,
                )
                continue
            adapter = _adapter_for(args.kind, src, cfg)
            n = 0
            for chunk in adapter.iter_chunks():
                store.write_chunk(chunk)
                n += 1
            total += n
            print(f"[ingest-source] {args.kind}: {n} chunks from {src}")

        print(
            f"[ingest-source] {total} chunks ingested "
            f"(idempotent on chunk_id; NOT materialized — run the ingest "
            f"cycle to materialize + extract)"
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ingest-source failed: {exc}", file=sys.stderr)
        return 1
