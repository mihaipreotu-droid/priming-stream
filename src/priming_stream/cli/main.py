"""Priming Stream command-line interface — the outer shell of the `prime` command.

Subcommands operate relative to the current working directory: paths are
resolved via ``resolve_paths(cfg, project_root=Path.cwd())``. Any command
other than ``init`` requires an initialized graph and fails gracefully with
a clear message otherwise.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from priming_stream.cli import clean as _clean_cli
from priming_stream.cli import coldstart as _coldstart_cli
from priming_stream.cli import daemon as _daemon_cli
from priming_stream.cli import db_snapshot as _db_snapshot_cli
from priming_stream.cli import echoes as _echoes_cli
from priming_stream.cli import ingest_source as _ingest_source_cli
from priming_stream.cli import install
from priming_stream.cli import reconcile as _reconcile_cli
from priming_stream.cli import sleep_auto as _sleep_auto_cli
from priming_stream.cli import record_ops as _record_ops_cli
from priming_stream.cli import sample as _sample_cli
from priming_stream.cli import search as _search_cli
from priming_stream.cli import sleep as _sleep_cli
from priming_stream.cli import vec_index as _vec_index_cli
from priming_stream.core.config import load_config
from priming_stream.core.db import connect
from priming_stream.core.graph_repo import GraphRepo
from priming_stream.core.paths import (
    ProjectPaths,
    ensure_dirs,
    resolve_paths,
    migrate_qmd_corpus_to_corpus,
)
from priming_stream.core.schema import apply_migrations
from priming_stream.inspector.dashboard import generate_dashboard


def _paths() -> ProjectPaths:
    """Resolve runtime paths relative to the current working directory."""
    cfg = load_config()
    return resolve_paths(cfg, project_root=Path.cwd())


def _require_graph(paths: ProjectPaths) -> str | None:
    """Return an error message if the graph DB is missing, else None."""
    if not paths.graph_db.exists():
        return (
            f"no graph database at {paths.graph_db} — run 'prime init' first"
        )
    return None


# -- subcommands ----------------------------------------------------------


def _cmd_init(args: argparse.Namespace) -> int:
    cfg = load_config()
    paths = resolve_paths(cfg, project_root=Path.cwd())

    # v0.7-x-vec-index folder rename: if a legacy ``storage/qmd-corpus/``
    # exists and ``storage/corpus/`` doesn't, rename atomically. Run
    # BEFORE ensure_dirs so we don't create an empty corpus/ next to the
    # legacy qmd-corpus/. Idempotent.
    paths.storage_dir.mkdir(parents=True, exist_ok=True)
    if migrate_qmd_corpus_to_corpus(paths.storage_dir):
        print(
            "migrated storage/qmd-corpus/ -> storage/corpus/ "
            "(v0.7-x-vec-index rename)"
        )
    ensure_dirs(paths)

    conn = connect(paths.graph_db)
    try:
        apply_migrations(conn)
    finally:
        conn.close()

    # v0.7-x-vec-index: materialize the ChromaDB persist dir + empty
    # 'records' collection so downstream code can open the index without
    # a chicken-and-egg create-on-first-write. Lazy fastembed init means
    # no model download here.
    try:
        from priming_stream.integrations.vec_index import RecordsVecIndex
        RecordsVecIndex(paths.vec_index_dir, cfg.vec_index.model_name)
    except Exception as exc:  # noqa: BLE001 - non-fatal
        print(
            f"warning: vec_index init skipped ({exc.__class__.__name__}: {exc})"
        )

    print(f"initialized Priming Stream storage at {paths.storage_dir}")
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    paths = _paths()
    err = _require_graph(paths)
    if err:
        print(err, file=sys.stderr)
        return 1

    conn = connect(paths.graph_db)
    try:
        repo = GraphRepo(conn)
        record_count = len(repo.list_records())
        cycles = repo.list_sleep_cycles(limit=10_000)
    finally:
        conn.close()

    print("graph stats:")
    print(f"  records       {record_count}")
    print(f"  sleep cycles  {len(cycles)}")
    return 0


def _cmd_dashboard(args: argparse.Namespace) -> int:
    paths = _paths()
    err = _require_graph(paths)
    if err:
        print(err, file=sys.stderr)
        return 1

    out = Path(args.out) if args.out else paths.storage_dir / "inspector.html"
    written = generate_dashboard(paths.graph_db, out)
    print(f"dashboard written to {written}")
    return 0


# -- parser ---------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prime",
        description="Priming Stream — persistent associative memory for an AI agent.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="initialize storage and graph DB")
    p_init.set_defaults(func=_cmd_init)

    p_stats = sub.add_parser("stats", help="print graph summary statistics")
    p_stats.set_defaults(func=_cmd_stats)

    p_dash = sub.add_parser("dashboard", help="generate the HTML inspector")
    p_dash.add_argument("--out", default=None, help="output HTML path")
    p_dash.set_defaults(func=_cmd_dashboard)

    install.register(sub)
    _clean_cli.register(sub)
    _coldstart_cli.register(sub)
    _ingest_source_cli.register(sub)
    _daemon_cli.register(sub)
    _echoes_cli.register(sub)
    _sample_cli.register(sub)
    _search_cli.register(sub)
    _sleep_cli.register(sub)
    _sleep_auto_cli.register(sub)
    _reconcile_cli.register(sub)
    _vec_index_cli.register(sub)
    _record_ops_cli.register(sub)
    _db_snapshot_cli.register(sub)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success, non-zero on error."""
    # Record summaries carry diacritics + symbols (≤, →) that crash a
    # cp1252 Windows console; best-effort utf-8 for every subcommand.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
    parser = _build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
