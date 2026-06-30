"""``prime search`` — on-demand search over the records substrate.

Two channels, mirroring the MCP tools:

- ``--semantic`` (default) — vector similarity (``graph_search_records``);
  associative, register-tolerant, misses bare exact terms.
- ``--lexical`` — FTS5 BM25 keyword/term match (``graph_search_lexical``),
  with ``--mode and|or|phrase``; finds an exact term / name / citation that
  the semantic channel would miss.

Owner-invoked + manual: unlike the MCP path (where the model composes the
query from intent), here you type the query and pick the channel/mode.
Read-only over the substrate.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from priming_stream.core.config import load_config
from priming_stream.core.db import connect
from priming_stream.core.graph_repo import GraphRepo
from priming_stream.core.paths import resolve_paths
from priming_stream.graph_ops import graph_search_lexical, graph_search_records
from priming_stream.integrations.vec_index import RecordsVecIndex


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "search",
        help="search the records substrate (semantic or lexical)",
    )
    p.add_argument("query", help="search text")
    chan = p.add_mutually_exclusive_group()
    chan.add_argument(
        "--semantic", action="store_true",
        help="vector similarity search (default)",
    )
    chan.add_argument(
        "--lexical", action="store_true",
        help="FTS5 keyword/term match (exact terms, names, citations)",
    )
    p.add_argument(
        "--mode", choices=["and", "or", "phrase"], default="and",
        help="lexical only: term combination (default: and)",
    )
    p.add_argument("-k", type=int, default=10, help="max results (default 10)")
    p.set_defaults(func=_cmd_search)


def _cmd_search(args: argparse.Namespace) -> int:
    cfg = load_config()
    paths = resolve_paths(cfg, project_root=Path.cwd())
    if not paths.graph_db.exists():
        print(
            f"no graph database at {paths.graph_db} — run 'prime init' first",
            file=sys.stderr,
        )
        return 1

    conn = connect(paths.graph_db)
    try:
        repo = GraphRepo(conn)
        if args.lexical:
            results = graph_search_lexical(args.query, args.k, args.mode, repo)
            channel = f"lexical/{args.mode}"
        else:
            vec = RecordsVecIndex(paths.vec_index_dir, cfg.vec_index.model_name)
            results = graph_search_records(args.query, args.k, vec, repo)
            channel = "semantic"
    finally:
        conn.close()

    print(f"[{channel}] {args.query!r} — {len(results)} hit(s)")
    for r in results:
        tag = r.get("source_date") or (
            "doc" if r.get("kind") == "index_card" else r.get("kind") or ""
        )
        score = r.get("score")
        score_s = f"{score:+.3f}" if isinstance(score, (int, float)) else "?"
        summary = (r.get("summary") or "").replace("\n", " ")[:160]
        print(f"  {r['record_id']}  score={score_s}  {tag}")
        print(f"    {summary}")
    return 0
