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
    p.add_argument(
        "--stats", action="store_true",
        help="per-day priming health: turns by source, empty-rate, "
             "client_ms percentiles incl. deadline breaches (P7)",
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


def _pctl(values: list, q: float):
    """Nearest-rank percentile; None on empty input."""
    if not values:
        return None
    vs = sorted(values)
    i = min(len(vs) - 1, max(0, int(round(q * (len(vs) - 1)))))
    return vs[i]


def _print_stats(echoes: list[dict]) -> None:
    """Per-day priming health (P7): one line per day, chronological.

    ``empty%`` is the headline counter — the number that stayed invisible
    for 5 days after the bge swap. ``client_ms`` is the hook-side wall time
    (uncensored: breaches included); ``>2s`` counts turns at/over the client
    deadline. Lines predating the P7 fields simply have no client_ms and are
    excluded from the timing columns (not from the counts).
    """
    by_day: dict[str, list[dict]] = {}
    for e in echoes:
        day = str(e.get("at") or "")[:10] or "?"
        by_day.setdefault(day, []).append(e)
    print(f"{'day':10s} {'turns':>5s} {'daemon':>6s} {'fallbk':>6s} "
          f"{'empty':>5s} {'empty%':>6s} {'p50ms':>6s} {'p90ms':>6s} "
          f"{'maxms':>6s} {'>2s':>4s} {'wNot':>5s} {'wFlr':>5s} {'wReg':>5s} "
          f"{'prv0%':>6s}")
    for day in sorted(by_day):
        es = by_day[day]
        n = len(es)
        src = {"daemon": 0, "fallback": 0, "empty": 0}
        # suffix-based: counts both the current whisper-* values and the
        # mute-* values from the brief pre-amendment enforcement window
        gate = {"notification": 0, "floor": 0, "regime": 0}
        for e in es:
            s = str(e.get("source") or "")
            if s in src:
                src[s] += 1
            g = str(e.get("gated") or "")
            for suffix in gate:
                if g.endswith(suffix):
                    gate[suffix] += 1
        cms = [e["client_ms"] for e in es
               if isinstance(e.get("client_ms"), (int, float))]
        breaches = sum(1 for v in cms if v >= 2000)
        p50, p90 = _pctl(cms, 0.5), _pctl(cms, 0.9)
        mx = max(cms) if cms else None
        # prv0%: of daemon turns that CARRY prev_len (post-P5 lines), the
        # fraction where the P5 response-seed came back empty. Watches the
        # silent-death mode of _last_assistant_text (a CC transcript schema
        # change would push this to 100% while every other column stays
        # green — 2026-07-21 review:. "—" = no post-P5 daemon lines that day.
        prevs = [e["prev_len"] for e in es
                 if e.get("source") == "daemon"
                 and isinstance(e.get("prev_len"), (int, float))]
        prev0 = (f"{100.0 * sum(1 for v in prevs if v == 0) / len(prevs):5.1f}%"
                 if prevs else f"{'—':>6s}")
        fmt = lambda v: f"{v:6.0f}" if isinstance(v, (int, float)) else f"{'—':>6s}"
        print(f"{day:10s} {n:5d} {src['daemon']:6d} {src['fallback']:6d} "
              f"{src['empty']:5d} {100.0 * src['empty'] / n:5.1f}% "
              f"{fmt(p50)} {fmt(p90)} {fmt(mx)} {breaches:4d} "
              f"{gate['notification']:5d} {gate['floor']:5d} "
              f"{gate['regime']:5d} {prev0}")


def _cmd_echoes(args: argparse.Namespace) -> int:
    cfg = load_config()
    paths = resolve_paths(cfg, project_root=Path.cwd())
    echoes_path = paths.episodic_dir / "echoes.jsonl"

    echoes = _read_echoes(echoes_path)
    if getattr(args, "stats", False):
        if args.session:
            echoes = [e for e in echoes
                      if args.session in (e.get("session_id") or "")]
        if not echoes:
            print(f"no echoes at {echoes_path}")
            return 0
        _print_stats(echoes)
        return 0
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
            # P7: hook-side wall time — present on every post-2026-07-21 line,
            # the only timing on fallback/empty turns (breach-visible).
            cms = e.get("client_ms")
            if isinstance(cms, (int, float)):
                ms_s += f" cl={cms:.0f}ms"
            # P2/P3 turn-gate provenance ("full" stays silent — it's the norm)
            g = e.get("gated")
            if g and g != "full":
                ms_s += f" [{g}]"
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
