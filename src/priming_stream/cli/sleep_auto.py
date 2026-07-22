"""``prime sleep-auto`` — unattended consolidation cycle (W automation).

Triggered by Windows Task Scheduler (nightly by default). Flow:

1. **Lock** — a lock file under storage prevents concurrent cycles (a manual
   ``/prime-ingest`` or a previous auto-run still going). Stale locks are taken over.
2. **Discovery** — enumerate Claude Code session transcripts under
   ``~/.claude/projects/**/*.jsonl``, keep only **settled** ones (not modified in
   the last ``--settled-minutes``, so an in-progress session — including this
   automation's own ``claude -p`` session — is skipped), minus excluded dirs.
3. **Ingest** — feed the settled sessions through ``ClaudeCodeAdapter`` into the
   episodic store (idempotent on ``chunk_id``: re-seeing an old session is a no-op;
   only new chunks land). NO materialize here — that's the cycle's job.
4. **Pending check** — count chunks past the materialize cursor. If zero, no-op
   (skip the LLM entirely) and exit clean.
5. **Run the cycle** — shell out to
   ``claude -p --dangerously-skip-permissions "/prime-ingest --all-pending"``: a fresh
   HEADLESS Claude session (authed by the long-lived ``CLAUDE_CODE_OAUTH_TOKEN``)
   runs the unified ingest skill (materialize → extract → reconcile → finalize).
   This is the ONLY non-deterministic / LLM step.

``--dry-run`` does 1–4 and reports, skipping the ingest write + the LLM cycle —
for testing discovery without mutating anything.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from priming_stream.core import write_lock
from priming_stream.core.config import load_config
from priming_stream.core.episodic import EpisodicStore
from priming_stream.core.models import now_iso
from priming_stream.core.paths import ensure_dirs, resolve_paths
from priming_stream.ingest.claude_code import ClaudeCodeAdapter

_LOCK_NAME = write_lock.LOCK_NAME  # single source: core.write_lock
_LOG_NAME = "sleep_auto.log"
_DEFAULT_SETTLED_MIN = 30
_DEFAULT_LOCK_STALE_MIN = 180  # a cycle older than this → assume dead, take over


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "sleep-auto",
        help="unattended consolidation cycle (discover settled CC sessions, "
             "ingest, run /prime-ingest headless via claude -p). For Task Scheduler.",
    )
    p.add_argument(
        "--projects-dir", default=None,
        help="Claude Code projects root (default: ~/.claude/projects)",
    )
    p.add_argument(
        "--settled-minutes", type=int, default=_DEFAULT_SETTLED_MIN,
        help="skip sessions modified within the last N minutes (in-progress "
             f"guard; default {_DEFAULT_SETTLED_MIN})",
    )
    p.add_argument(
        "--exclude", action="append", default=None, dest="excludes",
        metavar="SUBSTR",
        help="skip session paths containing this substring (repeatable). The "
             "automation's own claude -p project dir should be excluded.",
    )
    p.add_argument(
        "--claude-cmd", default="claude",
        help="claude executable (default 'claude'; on Windows the .cmd is "
             "resolved via the shell)",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="cap chunks extracted per run (passed to /prime-ingest). Default = "
             "drain all pending. Use a cap to chew a large first-run backlog "
             "over several runs.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="discover + report only; do NOT ingest or run the LLM cycle",
    )
    p.set_defaults(func=cmd_sleep_auto)


# -- helpers --------------------------------------------------------------


def _projects_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg).expanduser()
    return Path.home() / ".claude" / "projects"


def _log(paths, msg: str) -> None:
    line = f"[{now_iso()}] {msg}"
    print(line)
    try:
        with (paths.storage_dir / _LOG_NAME).open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def _settled_sessions(
    projects_dir: Path, settled_minutes: int, excludes: list[str],
) -> list[Path]:
    """``.jsonl`` transcripts not touched in the last ``settled_minutes``,
    excluding any path containing an ``excludes`` substring."""
    if not projects_dir.is_dir():
        return []
    cutoff = time.time() - settled_minutes * 60
    out: list[Path] = []
    for p in projects_dir.rglob("*.jsonl"):
        # skip sub-agent transcripts (`<session>/subagents/agent-*.jsonl`) —
        # those are internal Workflow/Task workers, not conversations.
        if "subagents" in p.parts:
            continue
        s = str(p)
        if any(x and x in s for x in excludes):
            continue
        try:
            if p.stat().st_mtime <= cutoff:
                out.append(p)
        except OSError:
            continue
    return sorted(out)


def _pending_count(paths, cfg) -> int:
    """Chunks past the materialize cursor (what a cycle would extract)."""
    import json

    cursor_path = paths.corpus_cursor_path
    last_seen = None
    if cursor_path.exists():
        try:
            last_seen = json.loads(cursor_path.read_text(encoding="utf-8")).get(
                "last_chunk_id"
            )
        except (OSError, ValueError):
            last_seen = None
    store = EpisodicStore(paths.episodic_dir)
    skipping = last_seen is not None
    n = 0
    for chunk in store.iter_chunks():
        if skipping:
            if chunk.chunk_id == last_seen:
                skipping = False
            continue
        n += 1
    # cursor referenced an unseen id (rewritten log) → count nothing rather
    # than everything (mirrors materialize_pending's conservative stance).
    return 0 if skipping and last_seen is not None else n


# 2026-07-21 review: the old exists()-then-write lockfile had a seconds-
# wide TOCTOU window; the lock is now a REAL OS file lock in
# ``core.write_lock`` (atomic acquire, auto-release on process death — the
# stale-age / PID-liveness heuristics went with the race). The handle lives
# in a module global for the duration of the cycle; ``stale_minutes`` is
# kept in the signature for existing callers but is unused.
_LOCK_HANDLE: object | None = None


def _acquire_lock(paths, stale_minutes: int = 0) -> bool:  # noqa: ARG001
    global _LOCK_HANDLE
    handle = write_lock.acquire(paths.storage_dir, holder_note=now_iso())
    if handle is None:
        return False
    _LOCK_HANDLE = handle
    return True


def _release_lock(paths) -> None:  # noqa: ARG001 - signature kept for callers
    global _LOCK_HANDLE
    handle = _LOCK_HANDLE
    _LOCK_HANDLE = None
    write_lock.release(handle)


# -- command --------------------------------------------------------------


def cmd_sleep_auto(args: argparse.Namespace) -> int:
    cfg = load_config()
    paths = resolve_paths(cfg, project_root=Path.cwd())
    if not paths.graph_db.exists():
        print(
            f"no graph database at {paths.graph_db} — run 'prime init' first",
            file=sys.stderr,
        )
        return 1
    ensure_dirs(paths)

    excludes = list(args.excludes or [])

    if not args.dry_run and not _acquire_lock(paths, _DEFAULT_LOCK_STALE_MIN):
        _log(paths, "lock held by a live cycle — skipping this run")
        return 0
    try:
        proj = _projects_dir(args.projects_dir)
        sessions = _settled_sessions(proj, args.settled_minutes, excludes)
        _log(paths, f"discovery: {len(sessions)} settled CC sessions under {proj}")

        if args.dry_run:
            for s in sessions[:20]:
                _log(paths, f"  would ingest: {s}")
            _log(paths, f"pending chunks (current cursor): {_pending_count(paths, cfg)}")
            _log(paths, "dry-run: no ingest, no cycle")
            return 0

        # ingest settled sessions (idempotent on chunk_id; no materialize)
        store = EpisodicStore(paths.episodic_dir)
        ingested = 0
        for s in sessions:
            try:
                for chunk in ClaudeCodeAdapter(
                    s,
                    idle_minutes=cfg.sleep.idle_minutes,
                    chunk_max_turns=cfg.sleep.chunk_max_turns,
                ).iter_chunks():
                    store.write_chunk(chunk)
                    ingested += 1
            except Exception as exc:  # noqa: BLE001 - one bad transcript ≠ abort
                _log(paths, f"  WARN ingest failed for {s.name}: {exc}")
        _log(paths, f"ingest: {ingested} chunks attempted (idempotent on id)")

        pending = _pending_count(paths, cfg)
        if pending == 0:
            _log(paths, "nothing pending — no LLM cycle this run")
            return 0
        _log(paths, f"{pending} pending chunks → running headless /prime-ingest")

        # the only LLM step: a fresh headless Claude session runs the skill.
        # cwd = the project dir so the skill's hardcoded paths + the project
        # config resolve.
        scope = str(args.limit) if args.limit else "--all-pending"
        full_path = shutil.which(args.claude_cmd)
        if full_path is None:
            _log(paths, f"ERROR: claude executable not found: {args.claude_cmd!r}")
            return 1
        # On Windows, .cmd/.bat wrappers require cmd /c to invoke correctly.
        if full_path.lower().endswith((".cmd", ".bat")):
            argv = ["cmd", "/c", full_path, "-p",
                    "--dangerously-skip-permissions", f"/prime-ingest {scope}"]
        else:
            argv = [full_path, "-p", "--dangerously-skip-permissions",
                    f"/prime-ingest {scope}"]
        proc = subprocess.run(
            argv, shell=False, cwd=str(Path.cwd()),
            stdin=subprocess.DEVNULL,  # claude -p else waits 3s for piped stdin
            capture_output=True, text=True,
            # the child (claude -p) emits UTF-8 (RO diacritics, ≤, em-dash); the
            # default Windows console codec (cp1252) crashes the reader thread on
            # the first non-cp1252 byte, swallowing the whole tail. Decode UTF-8
            # explicitly and never let an undecodable byte kill the capture.
            encoding="utf-8", errors="replace",
            timeout=3600,
        )
        tail = (proc.stdout or "").strip().splitlines()[-3:]
        _log(paths, f"cycle exit={proc.returncode}; tail: {' | '.join(tail)}")
        if proc.returncode != 0:
            err = (proc.stderr or "").strip().splitlines()[-3:]
            _log(paths, f"  stderr: {' | '.join(err)}")
            return proc.returncode
        return 0
    finally:
        if not args.dry_run:
            _release_lock(paths)
