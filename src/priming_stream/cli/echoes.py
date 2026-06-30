"""``prime echoes`` — E.1 memory-echoes review surface (§16.5).

Reads ``storage/episodic/echoes.jsonl`` — written by the UserPromptSubmit
hook (one line per prompt: which records the substrate primed, via which
path, 30-day retention) — and renders recent echoes with record ids
resolved to summaries via SQLite at display time. The human half of the
"shared associative substrate" contract: see what the bridge primed, after the
fact, greppable and durable.

Read-only over the substrate; the only writer of echoes.jsonl is the hook.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from priming_stream.core.config import load_config
from priming_stream.core.db import connect
from priming_stream.core.graph_repo import GraphRepo
from priming_stream.core.paths import resolve_paths
from priming_stream.core.usage_join import (
    attach_usage_to_echoes,
    classify_usage,
    read_usage,
    role_for,
)


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "echoes",
        help="show what the substrate primed into recent turns (E.1)",
    )
    p.add_argument(
        "--last", type=int, default=5,
        help="how many recent echoes to show (default 5)",
    )
    p.add_argument(
        "--session", default=None,
        help="filter: session_id substring match",
    )
    p.add_argument(
        "--ids-only", action="store_true",
        help="skip summary resolution (raw ids, faster)",
    )
    p.add_argument(
        "--no-usage", action="store_true",
        help="hide the active-use (usage.jsonl) block per turn",
    )
    p.set_defaults(func=_cmd_echoes)


def _read_echoes(path: Path) -> list[dict]:
    """Tolerant reader: skip blank/corrupt lines (same posture as episodic)."""
    out: list[dict] = []
    if not path.exists():
        return out
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
    return out


def _date_tag(record) -> str:
    if record.source_date:
        return record.source_date.replace("T", " ").rstrip("Z")[:16]
    return "doc" if getattr(record, "kind", "") == "index_card" else "manual"


def _print_used(echo: dict, repo) -> None:
    """Print the active-use block for one turn (usage.jsonl entries joined).

    One line per MCP read attributed to this turn, tagged with how it relates
    to what the turn surfaced (verified-use / recall-miss / …). Silent when the
    turn has no recorded active use.
    """
    used = echo.get("used") or []
    if not used:
        return
    print("  used (active):")
    for u in used:
        tag = classify_usage(u, echo)
        role = role_for(u.get("tool", ""))
        tool = u.get("tool", "?")
        if role == "fetch":
            rid = u.get("record_id") or "?"
            detail = rid
            if repo is not None and rid != "?":
                rec = repo.get_record(rid)
                if rec is not None and rec.summary:
                    detail = f"{rid} {(rec.summary).replace(chr(10), ' ')[:80]}"
            print(f"    [{tag}] {tool} {detail}")
        else:
            q = (u.get("query") or "").replace("\n", " ")[:80]
            n = len(u.get("result_ids") or [])
            mode = f" mode={u['mode']}" if u.get("mode") else ""
            print(f"    [{tag}] {tool}{mode} \"{q}\" → {n} hit(s)")


def _cmd_echoes(args: argparse.Namespace) -> int:
    cfg = load_config()
    paths = resolve_paths(cfg, project_root=Path.cwd())
    echoes_path = paths.episodic_dir / "echoes.jsonl"

    echoes = _read_echoes(echoes_path)
    # Active-use join (E.1 sibling): attach usage.jsonl entries to the turn
    # that primed them BEFORE filtering/slicing, so the join sees every echo.
    # getattr keeps callers that build the Namespace by hand (older tests)
    # working without the flag.
    if not getattr(args, "no_usage", False):
        usage = read_usage(paths.episodic_dir / "usage.jsonl")
        echoes, _orphans = attach_usage_to_echoes(echoes, usage)
    if args.session:
        echoes = [e for e in echoes if args.session in (e.get("session_id") or "")]
    if not echoes:
        print(f"no echoes at {echoes_path}" + (
            f" matching session {args.session!r}" if args.session else ""
        ))
        return 0
    echoes = echoes[-max(args.last, 1):]

    repo = None
    conn = None
    if not args.ids_only and paths.graph_db.exists():
        conn = connect(paths.graph_db)
        repo = GraphRepo(conn)

    try:
        for e in echoes:
            sess = (e.get("session_id") or "")[:8] or "?"
            ms = e.get("spread_ms")
            ms_s = f" {ms:.0f}ms" if isinstance(ms, (int, float)) else ""
            print(
                f"{e.get('at', '?')}  sess={sess}  "
                f"{e.get('source', '?')}{ms_s}  "
                f"\"{e.get('prompt_head', '')}\""
            )
            for bucket, label in (("semantic", "A"), ("lexical", "B")):
                for rid in e.get(bucket) or []:
                    if repo is None:
                        print(f"  {label} {rid}")
                        continue
                    rec = repo.get_record(rid)
                    if rec is None:
                        print(f"  {label} {rid} (deleted)")
                        continue
                    summary = (rec.summary or "").replace("\n", " ")[:120]
                    print(f"  {label} {rid} [{_date_tag(rec)}] {summary}")
            _print_used(e, repo)
            print()
    finally:
        if conn is not None:
            conn.close()
    return 0
