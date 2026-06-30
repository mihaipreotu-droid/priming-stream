"""``prime record create|edit|delete|restore`` — owner-curation of records.

Waking-time CRUD on a single record — the owner's deliberate write channel,
and (with the sleep cycle) one of only two paths that mutate the durable
substrate. MCP stays strictly read-only by design (arch §5.1/§11); curation
is CLI-only, explicit and visible. This is the owner authoring / correcting /
removing his own records — the "editable" leg of the inspectable-cognitive-
extension contract.

``create`` makes an owner-authored ``claim``: ground truth, not an LLM
distillation, so it has no episodic chunk to verify against. It is anchored
like an ``index_card`` (``source_uri = owner://``, offsets 0/0 — no fabricated
chunk anchor) and stamped with ``source_date = creation time`` so it ages in
A.5b recency like any conversational record (a NULL ``source_date`` would
freeze ``f_recency`` at 1 forever).

SQL-canonical (2026-06-12): the ``records`` table in ``graph.db`` is the
source of truth; ChromaDB is a derived cache rebuilt from it
(``vec-index-rebuild``). Every mutation writes SQLite first (the FTS5 shadow
syncs via triggers), then best-effort re-embeds ChromaDB, then nudges the
daemon. A failed vec write never blocks the command — ``vec-index-rebuild``
recovers it.

Soft delete (default) moves the row to the ``records_trash`` table —
reversible at zero cost via ``record restore <id>``. ``--hard`` deletes the
row outright (recoverable only from a DB snapshot).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from priming_stream.core.config import load_config
from priming_stream.core.db import connect
from priming_stream.core.graph_repo import GraphRepo
from priming_stream.core.models import Record, new_record_id, now_iso
from priming_stream.core.paths import ProjectPaths, resolve_paths
from priming_stream.core.schema import apply_migrations
from priming_stream.integrations.vec_index import RecordsVecIndex


# -- registration ---------------------------------------------------------


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "record",
        help="create, edit, delete or restore a single record (owner curation)",
    )
    rsub = p.add_subparsers(dest="record_command", required=True)

    pc = rsub.add_parser(
        "create",
        help="create a new owner-authored claim record (live — searchable "
             "on the next bridge fire, no sleep wait)",
    )
    pc.add_argument(
        "text", nargs="?", default=None,
        help="record summary text (or pass --summary-file for long / "
             "quoted text)",
    )
    pc.add_argument(
        "--summary-file", default=None,
        help="path to a UTF-8 file holding the summary",
    )
    pc.set_defaults(func=_cmd_create)

    pe = rsub.add_parser(
        "edit",
        help="replace a record's summary across SQLite + vec_index",
    )
    pe.add_argument("record_id", help="record id (rec_xxxxxxxx)")
    grp = pe.add_mutually_exclusive_group(required=True)
    grp.add_argument("--summary", default=None, help="new summary text")
    grp.add_argument(
        "--summary-file", default=None,
        help="path to a UTF-8 file holding the new summary (robust for "
             "long / quoted text)",
    )
    pe.set_defaults(func=_cmd_edit)

    pd = rsub.add_parser(
        "delete",
        help="remove a record from SQLite + vec_index",
    )
    pd.add_argument("record_id", help="record id (rec_xxxxxxxx)")
    pd.add_argument(
        "--hard", action="store_true",
        help="delete the row outright (default: move it to the "
             "records_trash table so the delete is reversible via "
             "'record restore')",
    )
    pd.set_defaults(func=_cmd_delete)

    pr = rsub.add_parser(
        "restore",
        help="reverse a soft delete: move a record back from records_trash "
             "into the live substrate (+ re-embed)",
    )
    pr.add_argument("record_id", help="record id (rec_xxxxxxxx)")
    pr.set_defaults(func=_cmd_restore)


# -- helpers --------------------------------------------------------------


def _paths() -> ProjectPaths:
    cfg = load_config()
    return resolve_paths(cfg, project_root=Path.cwd())


def _require_graph(paths: ProjectPaths) -> str | None:
    if not paths.graph_db.exists():
        return f"no graph database at {paths.graph_db} — run 'prime init' first"
    return None


def _reload_daemon() -> None:
    """Best-effort: nudge the resident daemon to re-read the vec_index.
    Never fails the command."""
    try:
        from priming_stream.daemon import client as daemon_client

        res = daemon_client.reload_daemon(timeout_s=5.0)
        if res is None:
            print("  daemon: not running (change applies on next start)")
        else:
            print(
                f"  daemon: reloaded "
                f"({res.get('records_after', '?')} records)"
            )
    except Exception as exc:  # noqa: BLE001 - reload is non-fatal
        print(f"  daemon: reload failed ({exc})", file=sys.stderr)


def _resolve_summary(args: argparse.Namespace) -> str:
    if args.summary_file:
        text = Path(args.summary_file).read_text(encoding="utf-8")
    else:
        text = args.summary or ""
    return text.strip()


def _vec_embed(paths: ProjectPaths, cfg, record_id: str, summary: str) -> None:
    """Best-effort ChromaDB upsert — SQLite is the source of truth and the
    vec_index is rebuildable, so a failure is reported, never fatal."""
    try:
        vec = RecordsVecIndex(paths.vec_index_dir, cfg.vec_index.model_name)
        vec.add_record(record_id, summary)
    except Exception as exc:  # noqa: BLE001 - vec_index is rebuildable
        print(
            f"  vec_index embed failed ({exc.__class__.__name__}); "
            f"recover via 'prime vec-index-rebuild'",
            file=sys.stderr,
        )


def _vec_delete(paths: ProjectPaths, cfg, record_id: str) -> None:
    """Best-effort ChromaDB delete (same posture as ``_vec_embed``)."""
    try:
        vec = RecordsVecIndex(paths.vec_index_dir, cfg.vec_index.model_name)
        vec.delete_record(record_id)
    except Exception as exc:  # noqa: BLE001 - vec_index is rebuildable
        print(
            f"  vec_index delete failed ({exc.__class__.__name__}); "
            f"recover via 'prime vec-index-rebuild'",
            file=sys.stderr,
        )


# -- commands -------------------------------------------------------------


def _cmd_create(args: argparse.Namespace) -> int:
    if args.summary_file:
        text = Path(args.summary_file).read_text(encoding="utf-8").strip()
    else:
        text = (args.text or "").strip()
    if not text:
        print(
            "create: summary is empty (pass text or --summary-file)",
            file=sys.stderr,
        )
        return 1

    cfg = load_config()
    paths = resolve_paths(cfg, project_root=Path.cwd())
    err = _require_graph(paths)
    if err:
        print(err, file=sys.stderr)
        return 1

    # Owner-authored ground truth: anchored like an index_card (no episodic
    # chunk to verify against -> owner://, 0/0) and stamped source_date =
    # created_at so A.5b recency ages it like any conversational record.
    ts = now_iso()
    record = Record(
        id=new_record_id(),
        source_uri="owner://",
        anchor_offset_start=0,
        anchor_offset_end=0,
        summary=text,
        created_at=ts,
        kind="claim",
        source_date=ts,
    )

    # 1. SQLite (source of truth; records_ai trigger syncs the FTS5 shadow).
    conn = connect(paths.graph_db)
    try:
        apply_migrations(conn)
        GraphRepo(conn).create_record(record)
    finally:
        conn.close()

    # 2. ChromaDB embed — best-effort, rebuildable.
    _vec_embed(paths, cfg, record.id, record.summary)

    # 3. Daemon.
    _reload_daemon()

    print(f"created {record.id}")
    print(f"  {record.summary}")
    return 0


def _cmd_edit(args: argparse.Namespace) -> int:
    new_summary = _resolve_summary(args)
    if not new_summary:
        print("edit: new summary is empty", file=sys.stderr)
        return 1

    cfg = load_config()
    paths = resolve_paths(cfg, project_root=Path.cwd())
    err = _require_graph(paths)
    if err:
        print(err, file=sys.stderr)
        return 1

    conn = connect(paths.graph_db)
    try:
        apply_migrations(conn)
        repo = GraphRepo(conn)
        rec = repo.get_record(args.record_id)
        if rec is None:
            print(f"edit: no record {args.record_id!r}", file=sys.stderr)
            return 1
        if rec.kind != "claim":
            print(
                f"edit: {rec.id} is a {rec.kind!r}, not a claim — index_card "
                f"summaries are regenerated on content_hash change and an "
                f"edit would be clobbered. Refusing.",
                file=sys.stderr,
            )
            return 1

        old_summary = rec.summary

        # 1. SQLite (source of truth; FTS5 auto-synced by records_au).
        repo.update_record_summary(rec.id, new_summary)
    finally:
        conn.close()

    # 2. ChromaDB re-embed (upsert recomputes the vector from new text).
    _vec_embed(paths, cfg, rec.id, new_summary)

    # 3. Daemon.
    _reload_daemon()

    print(f"edited {rec.id}")
    print(f"  old: {old_summary}")
    print(f"  new: {new_summary}")
    return 0


def _cmd_delete(args: argparse.Namespace) -> int:
    cfg = load_config()
    paths = resolve_paths(cfg, project_root=Path.cwd())
    err = _require_graph(paths)
    if err:
        print(err, file=sys.stderr)
        return 1

    conn = connect(paths.graph_db)
    try:
        apply_migrations(conn)
        repo = GraphRepo(conn)
        rec = repo.get_record(args.record_id)
        if rec is None:
            print(f"delete: no record {args.record_id!r}", file=sys.stderr)
            return 1

        old_summary = rec.summary

        # 1. SQLite (source of truth; records_ad trigger drops the FTS5 row).
        if args.hard:
            repo.delete_record(rec.id)
            disposition = "row deleted"
        else:
            repo.trash_record(rec.id, reason="owner-delete")
            disposition = "moved to records_trash"
    finally:
        conn.close()

    # 2. ChromaDB.
    _vec_delete(paths, cfg, rec.id)

    # 3. Daemon.
    _reload_daemon()

    print(f"deleted {rec.id} ({'hard' if args.hard else 'soft'}; {disposition})")
    print(f"  was: {old_summary}")
    return 0


def _cmd_restore(args: argparse.Namespace) -> int:
    cfg = load_config()
    paths = resolve_paths(cfg, project_root=Path.cwd())
    err = _require_graph(paths)
    if err:
        print(err, file=sys.stderr)
        return 1

    conn = connect(paths.graph_db)
    try:
        apply_migrations(conn)
        repo = GraphRepo(conn)
        rec = repo.restore_record(args.record_id)
        if rec is None:
            print(
                f"restore: no record {args.record_id!r} in trash "
                f"(or its id is already live)",
                file=sys.stderr,
            )
            return 1
    finally:
        conn.close()

    # Re-embed the restored record so it is searchable again.
    _vec_embed(paths, cfg, rec.id, rec.summary)
    _reload_daemon()

    print(f"restored {rec.id}")
    print(f"  {rec.summary}")
    return 0
