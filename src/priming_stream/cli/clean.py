"""``prime clean-scratch`` — remove the ingest cycle's temporary artifacts.

The ingest skill drops a number of per-cycle scratch files/dirs under
``storage/corpus/`` (planner manifests, per-conversation assignment + results
dirs, the doc-branch work dir + markitdown conversions, the produced-docs
handoff, the reconcile plan). They are all **regenerated each cycle**, so they
are safe to delete after a run — keeping ``storage/`` tidy.

DELIBERATELY does NOT touch durable state: ``_cursor.json`` (materialize
position), ``records/`` / ``imports/`` (the substrate + episodic source), any
``_pre_*_snapshot`` rollback dir. Only the explicit allowlist below is removed.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from priming_stream.core.config import load_config
from priming_stream.core.paths import resolve_paths

# Per-cycle scratch under storage/corpus/ — files and dirs. Explicit allowlist
# (NOT a glob wipe) so durable state can never be caught by accident.
_SCRATCH = [
    "_sleep_manifest.json",
    "_sleep_index.json",
    "_sleep_assign",
    "_sleep_results",
    "_doc_index.json",
    "_doc_assign",
    "_doc_results",
    "_doc_md",
    "_produced_docs.json",
    "_reconcile_plan.json",
    "_claim_reconcile_plan.json",
    "_judge_batches",
    "_ingest_cycle.json",
]


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "clean-scratch",
        help="remove the ingest cycle's regenerable temp artifacts from "
             "storage/corpus (keeps cursor/records/imports/snapshots)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="list what would be removed without deleting",
    )
    p.set_defaults(func=cmd_clean_scratch)

    pc = subparsers.add_parser(
        "clean-cc-subagents",
        help="prune OLD Claude Code sub-agent transcripts "
             "(~/.claude/projects/**/subagents/agent-*.jsonl). Dry-run unless "
             "--execute; never touches main session transcripts.",
    )
    pc.add_argument(
        "--older-than", type=int, required=True, metavar="DAYS",
        help="only prune sub-agent transcripts older than this many days",
    )
    pc.add_argument(
        "--projects-dir", default=None,
        help="Claude Code projects root (default: ~/.claude/projects)",
    )
    pc.add_argument(
        "--execute", action="store_true",
        help="actually delete (default is a dry-run preview)",
    )
    pc.set_defaults(func=cmd_clean_cc_subagents)


def cmd_clean_scratch(args: argparse.Namespace) -> int:
    cfg = load_config()
    paths = resolve_paths(cfg, project_root=Path.cwd())
    corpus = paths.corpus_dir
    removed = []
    for name in _SCRATCH:
        target = corpus / name
        if not target.exists():
            continue
        if args.dry_run:
            removed.append(name)
            continue
        try:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            removed.append(name)
        except OSError as exc:
            print(f"  WARN could not remove {name}: {exc}", file=sys.stderr)
    verb = "would remove" if args.dry_run else "removed"
    print(f"clean-scratch: {verb} {len(removed)} item(s)"
          + (f": {', '.join(removed)}" if removed else ""))
    return 0


def cmd_clean_cc_subagents(args: argparse.Namespace) -> int:
    """Prune OLD sub-agent transcripts. Double-guarded so a main session
    transcript can NEVER be deleted: the file must (a) sit under a
    ``subagents`` directory AND (b) be named ``agent-*.jsonl``."""
    root = (
        Path(args.projects_dir).expanduser() if args.projects_dir
        else Path.home() / ".claude" / "projects"
    )
    if not root.is_dir():
        print(f"clean-cc-subagents: no projects dir at {root}", file=sys.stderr)
        return 1

    cutoff = time.time() - args.older_than * 86400
    victims: list[Path] = []
    total_bytes = 0
    for p in root.rglob("*.jsonl"):
        if "subagents" not in p.parts or not p.name.startswith("agent-"):
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        if st.st_mtime <= cutoff:
            victims.append(p)
            total_bytes += st.st_size

    mb = total_bytes / (1024 * 1024)
    if not args.execute:
        print(f"clean-cc-subagents [DRY-RUN]: {len(victims)} sub-agent "
              f"transcript(s) older than {args.older_than}d ({mb:.1f} MB). "
              f"Re-run with --execute to delete.")
        return 0

    deleted = 0
    for p in victims:
        try:
            p.unlink()
            deleted += 1
        except OSError as exc:
            print(f"  WARN could not delete {p.name}: {exc}", file=sys.stderr)
    print(f"clean-cc-subagents: deleted {deleted} sub-agent transcript(s) "
          f"({mb:.1f} MB).")
    return 0
