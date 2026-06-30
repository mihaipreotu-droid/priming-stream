"""``prime sleep-prepare`` / ``prime sleep-finalize`` — CLI helpers
the ``/prime-ingest`` Claude Code skill shells out to.

The skill body (``.claude/skills/prime-ingest/SKILL.md``) is the agent-side
script. These two subcommands bracket it on the Python side:

- ``sleep-prepare`` drains pending chunks → .md, opens a ``sleep_cycles``
  row, and prints a JSON manifest the skill parses. No external indexer:
  the vec_index only carries records (not chunks); chunks live on disk
  and the bridge reaches them via ``graph_chunk_around_anchor`` (tier-2).
- ``sleep-finalize`` PROMOTES this cycle's staged rows
  (``records_staging``, written by the skill's bulk-writers) into the
  canonical ``records`` table AND the ChromaDB ``records`` collection,
  then closes the cycle row with metrics. Per-record vec_index failures
  are logged into ``metrics_json`` and do NOT crash the cycle (the
  SQLite promotion already succeeded; user can backfill via
  ``vec-index-rebuild``).

The agent itself handles the per-chunk filter + LLM extract + record
write loop in between, in-session — no subprocess spawn for the LLM
(see ARCHITECTURE.md).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from priming_stream.core.config import load_config
from priming_stream.core.db import connect
from priming_stream.core.episodic import EpisodicStore
from priming_stream.core.graph_repo import GraphRepo
from priming_stream.core.models import Record, now_iso
from priming_stream.core.paths import ProjectPaths, resolve_paths
from priming_stream.core.schema import apply_migrations
from priming_stream.core.source_date import ensure_source_dates_in_db
from priming_stream.core.source_uri import (
    _AUTHORITY_IMPORTS,
    build as build_uri,
)
from priming_stream.ingest.materialize import (
    materialize_pending,
    save_pending_cursor,
)
from priming_stream.integrations.vec_index import RecordsVecIndex


def register(subparsers) -> None:
    p_prep = subparsers.add_parser(
        "sleep-prepare",
        help="materialize pending chunks to .md; open a sleep_cycles row; "
             "emit JSON manifest on stdout",
    )
    g = p_prep.add_mutually_exclusive_group()
    g.add_argument(
        "--limit", type=int, default=None,
        help="cap pending chunks processed this cycle",
    )
    g.add_argument(
        "--all-pending", action="store_true",
        help="drain everything (default when --limit is omitted)",
    )
    p_prep.add_argument(
        "--no-materialize", action="store_true",
        help="open an EMPTY sleep_cycles row WITHOUT materializing or "
             "draining any pending conversation chunks (does NOT advance the "
             "cursor). Used by /prime-ingest to obtain a cycle id for reconciling "
             "index cards — materializing here would silently skip those "
             "chunks' conversational extraction.",
    )
    p_prep.set_defaults(func=cmd_sleep_prepare)

    p_fin = subparsers.add_parser(
        "sleep-finalize",
        help="reconcile records to SQLite + vec_index; close the "
             "sleep_cycles row with metrics",
    )
    p_fin.add_argument("--cycle-id", type=int, required=True)
    p_fin.add_argument("--chunks-materialized", type=int, default=0)
    p_fin.add_argument("--records-created", type=int, default=0)
    p_fin.add_argument("--records-skipped", type=int, default=0)
    p_fin.add_argument(
        "--notes", default=None,
        help="optional free-text notes stored on the sleep_cycles row",
    )
    p_fin.add_argument(
        "--skip-vec-index", action="store_true",
        help="skip vec_index writes (useful for tests / when index "
             "will be rebuilt separately)",
    )
    p_fin.add_argument(
        "--manifest", default=None, dest="manifest_path",
        help="path to the sleep manifest JSON produced by sleep-prepare "
             "(default: storage/corpus/_sleep_manifest.json). When present "
             "and non-empty, the cursor is committed here — after the SQLite "
             "reconcile — so a crash between prepare and finalize never loses "
             "materialized chunks silently. Absent / doc-only cycles: "
             "no cursor commit, no error.",
    )
    p_fin.set_defaults(func=cmd_sleep_finalize)


# -- helpers --------------------------------------------------------------


def _imports_relpath(path: Path, imports_root: Path) -> str:
    """Path of a materialized chunk relative to ``imports_root``, forward
    slashes (so it composes into a ``qmd://`` source_uri with the
    imports authority — see ``priming_stream.core.source_uri``).

    The ``qmd://`` URI scheme name is preserved verbatim per spec §1.4 /
    source_uri docstring — qmd is gone as a runtime dep, but the scheme
    is a stable historical name on the 155 existing records' frontmatter.
    """
    rel = path.resolve().relative_to(imports_root.resolve())
    return rel.as_posix()


# -- cursor crash-safety helper -------------------------------------------


def _commit_cursor_from_manifest(
    args: argparse.Namespace, paths,
) -> None:
    """Advance the materialize cursor from the manifest's prepared_chunks.

    Called by ``cmd_sleep_finalize`` AFTER the SQLite reconcile succeeds.
    Reads ``prepared_chunks`` from the manifest file (``args.manifest_path``
    if given, else the standard ``paths.corpus_sleep_manifest_path``).

    Semantics:
    - Manifest absent or unreadable → silent no-op (doc-only / --no-materialize
      cycles never produce a manifest with chunks; missing manifest after a crash
      before the manifest was written → same result).
    - ``prepared_chunks`` empty → no-op (empty or doc-only cycle).
    - Manifest ``cycle_id`` ≠ the cycle being finalized → loud no-op (a stale
      or newer manifest must never advance the cursor for this cycle).
    - Otherwise: reconstruct ``written`` as a list of ``Path`` objects (in
      manifest order, so the LAST path is the correct cursor target) and call
      ``save_pending_cursor`` — identical semantics to the old prepare-time
      commit.
    """
    manifest_path = Path(
        args.manifest_path
        if getattr(args, "manifest_path", None)
        else paths.corpus_sleep_manifest_path
    )
    if not manifest_path.exists():
        return  # no manifest → no-op (doc-only or crash before prepare wrote it)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return  # unreadable manifest → silent no-op
    m_cycle = manifest.get("cycle_id")
    a_cycle = getattr(args, "cycle_id", None)
    if m_cycle is not None and a_cycle is not None and int(m_cycle) != int(a_cycle):
        print(
            f"finalize: manifest cycle_id {m_cycle} != --cycle-id {a_cycle}; "
            "cursor not advanced",
            file=sys.stderr,
        )
        return
    prepared = manifest.get("prepared_chunks") or []
    if not prepared:
        return  # empty or doc-only cycle
    written = [Path(c["path"]) for c in prepared if c.get("path")]
    if not written:
        return
    save_pending_cursor(paths.corpus_cursor_path, written)


# -- records promotion (SQL-canonical) -------------------------------------


def _reconcile_records(
    repo: GraphRepo,
) -> tuple[dict, list[tuple[str, str]], list[str]]:
    """Promote this cycle's STAGED rows (``records_staging``) into the
    canonical ``records`` table.

    Two record kinds, two promote rules:

    - **claim** — append-only, keyed on ``id``. Idempotent INSERT-OR-IGNORE:
      an id already in ``records`` just clears its staged row. Promotion is
      one transaction per row (INSERT + staged DELETE — ``promote_record``),
      so a crash mid-promote re-runs cleanly on the leftover staged rows.
    - **index_card** — one card per ``doc_key``, keyed on ``doc_key`` (NOT
      id). Upsert by hash:
        * no card with this doc_key  -> promote (new vec item);
        * same ``content_hash``      -> skip (document unchanged), clear
          the staged row;
        * different ``content_hash`` -> regenerate: ``replace_record`` the
          old row with the staged one. The OLD id goes into
          ``vec_delete_ids`` so finalize drops its stale embedding; the NEW
          id goes into ``new_items``.

    Returns ``(metrics, new_items, vec_delete_ids)``:
      - ``metrics`` counts both kinds (``reconciled`` = claims+cards
        promoted; ``cards_created``/``cards_replaced``/``cards_unchanged``
        break out the index_card path);
      - ``new_items`` = ``[(rid, summary), ...]`` to ADD to vec_index;
      - ``vec_delete_ids`` = stale index_card ids to DELETE from vec_index.

    The staging table is the cycle boundary: the bulk-writers fill it, the
    reconcile applies judge deletions against it, and this promotion drains
    it. ``stage_record`` enforces at-most-one staged card per ``doc_key``
    (the old filename-keyed-by-doc_key invariant).
    """
    metrics = {
        "reconciled": 0,
        "skipped_existing": 0,
        "skipped_malformed": 0,
        "cards_created": 0,
        "cards_replaced": 0,
        "cards_unchanged": 0,
    }
    new_items: list[tuple[str, str]] = []
    vec_delete_ids: list[str] = []

    for rec in repo.list_staged():
        if rec.kind == "index_card":
            _reconcile_index_card(
                rec, repo, metrics, new_items, vec_delete_ids,
            )
            continue

        # -- claim path: INSERT OR IGNORE on id ----------------------------
        if repo.get_record(rec.id) is not None:
            metrics["skipped_existing"] += 1
            repo.delete_staged(rec.id)
            continue
        try:
            repo.promote_record(rec)
            metrics["reconciled"] += 1
            new_items.append((rec.id, rec.summary))
        except Exception:
            # Leave the staged row in place — re-seen (and retried) on the
            # next finalize, mirroring the old leave-the-.md behavior.
            metrics["skipped_malformed"] += 1
    return metrics, new_items, vec_delete_ids


def _reconcile_index_card(
    rec: Record,
    repo: GraphRepo,
    metrics: dict,
    new_items: list[tuple[str, str]],
    vec_delete_ids: list[str],
) -> None:
    """Upsert one staged index_card by ``doc_key`` (see ``_reconcile_records``).

    A card without a ``doc_key`` is malformed (the key is its identity) —
    trashed (reversible, with a reason) so it doesn't haunt every later
    cycle's staging scan.
    """
    if not rec.doc_key:
        repo.trash_staged(rec.id, reason="malformed-no-doc-key")
        metrics["skipped_malformed"] += 1
        return

    existing = repo.get_record_by_doc_key(rec.doc_key)
    if existing is None:
        try:
            repo.promote_record(rec)
            metrics["reconciled"] += 1
            metrics["cards_created"] += 1
            new_items.append((rec.id, rec.summary))
        except Exception:
            metrics["skipped_malformed"] += 1
        return

    # A card already exists for this document. Unchanged iff the source
    # content_hash matches (None == None counts as unchanged: two
    # file-less mention cards for the same doc_key are the same card).
    if (existing.content_hash or None) == (rec.content_hash or None):
        metrics["cards_unchanged"] += 1
        metrics["skipped_existing"] += 1
        repo.delete_staged(rec.id)
        return

    # Regenerate: drop the stale row + its embedding, insert the fresh one.
    # ALWAYS queue the old id for vec deletion, even when the regenerated
    # card reuses the same id. Reason: ``_write_to_vec_index`` skips
    # ``add_record`` for an id already in the index (the append-only claim
    # guard). Without an explicit delete first, a same-id replace would keep
    # the STALE embedding. Delete-then-add re-embeds correctly in both cases
    # (finalize drains vec_delete_ids before adding new_items).
    # replace_record performs DELETE+INSERT in a single transaction; the
    # staged DELETE after it is safe — a crash in between leaves a staged
    # row whose content_hash now matches the substrate card, so the re-run
    # lands in the cards_unchanged branch (idempotent).
    try:
        repo.replace_record(existing.id, rec)
        repo.delete_staged(rec.id)
        metrics["reconciled"] += 1
        metrics["cards_replaced"] += 1
        vec_delete_ids.append(existing.id)
        new_items.append((rec.id, rec.summary))
    except Exception:
        metrics["skipped_malformed"] += 1


def _write_to_vec_index(
    vec_index: RecordsVecIndex,
    items: list[tuple[str, str]],
) -> dict:
    """Push ``(rid, summary)`` pairs into vec_index, one at a time.

    Per-record try/except so a single fastembed/ChromaDB failure on one
    record doesn't lose the rest. Skips records already present (an
    idempotency guard for re-runs over the same .md files).

    Returns ``{added, already_present, errors}`` where ``errors`` is a
    list of ``{record_id, error}`` dicts.
    """
    metrics = {"added": 0, "already_present": 0, "errors": []}
    for rid, summary in items:
        try:
            if vec_index.has_record(rid):
                metrics["already_present"] += 1
                continue
            vec_index.add_record(rid, summary)
            metrics["added"] += 1
        except Exception as exc:  # noqa: BLE001 - per-record boundary
            metrics["errors"].append({
                "record_id": rid,
                "error": f"{exc.__class__.__name__}: {exc}",
            })
    return metrics


# -- sleep-prepare --------------------------------------------------------


def cmd_sleep_prepare(args: argparse.Namespace) -> int:
    try:
        cfg = load_config()
        paths = resolve_paths(cfg, project_root=Path.cwd())

        if not paths.graph_db.exists():
            print(
                f"no graph database at {paths.graph_db} — run 'prime init' first",
                file=sys.stderr,
            )
            return 1

        episodic = EpisodicStore(paths.episodic_dir)
        imports_root = paths.corpus_imports_dir
        cursor_path = paths.corpus_cursor_path

        limit = args.limit  # None when --all-pending or unset

        # Snapshot which chunk_ids correspond to the paths we materialize
        # this pass. ``materialize_pending`` returns absolute paths in
        # order; we reconstruct chunk_id from the filename stem (matches
        # the safe_filename convention used on write).
        #
        # --no-materialize: open an empty cycle without touching the cursor.
        # /prime-ingest needs a cycle id to reconcile index cards; draining
        # conversation chunks here would advance the cursor past chunks that
        # were never conversationally extracted (no conv workflow runs in the
        # ingest path) — silently losing them.
        if args.no_materialize:
            written = []
        else:
            written = materialize_pending(
                episodic, imports_root, cursor_path, limit=limit,
                save_cursor=False,
            )

        prepared_chunks: list[dict] = []
        for path in written:
            rel = _imports_relpath(path, imports_root)
            chunk_id = path.stem
            prepared_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "path": str(path),
                    "source_uri": build_uri(
                        "qmd",
                        collection=_AUTHORITY_IMPORTS,
                        path=rel,
                    ),
                }
            )

        # Cursor commit is intentionally DEFERRED to sleep-finalize (crash-safety).
        # The manifest written below carries prepared_chunks[].path so finalize
        # can reconstruct `written` and commit the cursor AFTER the SQLite
        # reconcile succeeds.  A crash between prepare and finalize therefore
        # leaves the cursor unchanged — the same chunks are re-materialized on
        # the next run (idempotent overwrite) and extraction is retried cleanly.
        # --no-materialize: cursor must not move (nothing was materialized).

        # Open a sleep_cycles row; the id round-trips to sleep-finalize
        # through the JSON manifest.
        conn = connect(paths.graph_db)
        try:
            repo = GraphRepo(conn)
            cycle_id = repo.start_sleep_cycle(started_at=now_iso())
        finally:
            conn.close()

        # In-place doc collections: §16.10 seam. v0.7-x panel leaves this
        # empty; we still emit the key so the skill can iterate
        # unconditionally.
        in_place_docs: list[str] = []

        manifest = {
            "cycle_id": cycle_id,
            "prepared_chunks": prepared_chunks,
            "in_place_docs": in_place_docs,
        }
        # Persist the manifest ourselves — finalize commits the cursor from
        # this file, so a direct-CLI prepare without stdout capture must
        # still leave it on disk (the skill's Out-File then just overwrites
        # it with identical content).
        manifest_json = json.dumps(manifest, ensure_ascii=False, indent=2)
        paths.corpus_sleep_manifest_path.parent.mkdir(
            parents=True, exist_ok=True,
        )
        paths.corpus_sleep_manifest_path.write_text(
            manifest_json, encoding="utf-8",
        )
        print(manifest_json)
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"sleep-prepare failed: {exc}", file=sys.stderr)
        return 1


# -- sleep-finalize -------------------------------------------------------


def cmd_sleep_finalize(args: argparse.Namespace) -> int:
    t0 = time.monotonic()
    try:
        cfg = load_config()
        paths = resolve_paths(cfg, project_root=Path.cwd())

        if not paths.graph_db.exists():
            print(
                f"no graph database at {paths.graph_db} — run 'prime init' first",
                file=sys.stderr,
            )
            return 1

        # Read the row's started_at + chunks-materialized estimate by
        # listing recent cycles. Simpler than adding a get-by-id repo
        # method; sleep cycles are bounded.
        conn = connect(paths.graph_db)
        try:
            repo = GraphRepo(conn)
            cycles = repo.list_sleep_cycles(limit=50)
        finally:
            conn.close()

        row = next((c for c in cycles if c["id"] == args.cycle_id), None)
        if row is None:
            print(
                f"sleep cycle id {args.cycle_id} not found",
                file=sys.stderr,
            )
            return 1
        if row["completed_at"]:
            print(
                f"sleep cycle id {args.cycle_id} already completed at "
                f"{row['completed_at']}",
                file=sys.stderr,
            )
            return 1

        # F-1: promote this cycle's staged rows into the canonical
        # ``records`` table. The skill's bulk-writers fill ``records_staging``
        # only; without this step, records are invisible to the bridge
        # (spreading + graph_search_records both read ``records``).
        # Idempotent — re-running is safe (INSERT-OR-IGNORE semantics for
        # claims, hash-checked upsert for cards, leftover staged rows retry).
        #
        # v0.7-x-B: derive source_date onto any staged row missing it (the
        # new records this cycle — the bulk-writer stages them without it).
        # Deterministic Python (source_uri+anchor over the episodic exports),
        # no LLM. Runs BEFORE the promotion so promoted rows carry the date.
        conn = connect(paths.graph_db)
        try:
            apply_migrations(conn)  # staging/trash tables exist (idempotent)
            sd_metrics = ensure_source_dates_in_db(
                conn,
                storage_dir=paths.storage_dir,
                corpus_dir=paths.corpus_dir,
            )
            repo = GraphRepo(conn)
            recon_metrics, new_items, vec_delete_ids = _reconcile_records(repo)
        finally:
            conn.close()

        # Crash-safety: cursor commit AFTER SQLite reconcile succeeded.
        # sleep-prepare no longer commits the cursor; instead it embeds the
        # prepared chunk paths in the manifest.  We read those paths here and
        # advance the cursor now — after reconcile — so a crash between prepare
        # and finalize leaves the cursor untouched and the same chunks are
        # safely re-materialized (idempotent) on the next run.
        #
        # Tolerant by design: manifest absent / prepared_chunks empty / doc-only
        # cycle (--no-materialize, which skips materialization entirely) → silent
        # no-op, cursor stays where it was.
        _commit_cursor_from_manifest(args, paths)

        # Push freshly-reconciled records into the vec_index, and drop the
        # embeddings of any index_card rows that were regenerated this cycle
        # (their old id is dead). Per spec §2.D2: failures are logged into
        # metrics (errors list), the cycle continues, exit code stays 0.
        vec_metrics: dict = {
            "added": 0,
            "already_present": 0,
            "deleted": 0,
            "errors": [],
            "skipped": False,
        }
        if args.skip_vec_index:
            vec_metrics["skipped"] = True
        elif new_items or vec_delete_ids:
            try:
                vec_index = RecordsVecIndex(
                    paths.vec_index_dir, cfg.vec_index.model_name,
                )
            except Exception as exc:  # noqa: BLE001 - open boundary
                vec_metrics["errors"].append({
                    "record_id": "<open>",
                    "error": f"{exc.__class__.__name__}: {exc}",
                })
            else:
                # Delete stale embeddings first (regenerated index cards),
                # then add the fresh ones. Per-id try/except so one failure
                # doesn't lose the rest.
                for dead_id in vec_delete_ids:
                    try:
                        vec_index.delete_record(dead_id)
                        vec_metrics["deleted"] += 1
                    except Exception as exc:  # noqa: BLE001 - per-id boundary
                        vec_metrics["errors"].append({
                            "record_id": dead_id,
                            "error": f"{exc.__class__.__name__}: {exc}",
                        })
                metrics = _write_to_vec_index(vec_index, new_items)
                vec_metrics["added"] = metrics["added"]
                vec_metrics["already_present"] = metrics["already_present"]
                vec_metrics["errors"].extend(metrics["errors"])

        elapsed_s = round(time.monotonic() - t0, 3)
        metrics = {
            "chunks_materialized": args.chunks_materialized,
            "records_created": args.records_created,
            "records_skipped": args.records_skipped,
            "records_reconciled": recon_metrics["reconciled"],
            "records_reconcile_skipped_existing":
                recon_metrics["skipped_existing"],
            "records_reconcile_skipped_malformed":
                recon_metrics["skipped_malformed"],
            "index_cards_created": recon_metrics["cards_created"],
            "index_cards_replaced": recon_metrics["cards_replaced"],
            "index_cards_unchanged": recon_metrics["cards_unchanged"],
            "source_date_written": sd_metrics["written"],
            "source_date_skipped_existing": sd_metrics["skipped_existing"],
            "source_date_no_date": sd_metrics["no_date"],
            "vec_index_added": vec_metrics["added"],
            "vec_index_already_present": vec_metrics["already_present"],
            "vec_index_deleted": vec_metrics["deleted"],
            "vec_index_errors": vec_metrics["errors"],
            "vec_index_skipped": vec_metrics["skipped"],
            "elapsed_s": elapsed_s,
        }
        completed_at = now_iso()

        conn = connect(paths.graph_db)
        try:
            repo = GraphRepo(conn)
            repo.finish_sleep_cycle(
                args.cycle_id,
                completed_at=completed_at,
                chunks_materialized=args.chunks_materialized,
                records_created=args.records_created,
                records_skipped=args.records_skipped,
                metrics_json=json.dumps(metrics, ensure_ascii=False),
                notes=args.notes,
            )
        finally:
            conn.close()

        print(
            json.dumps(
                {
                    "cycle_id": args.cycle_id,
                    "completed_at": completed_at,
                    "metrics": metrics,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

        # v0.7-x-daemon-reload: nudge the resident daemon to re-read the
        # records substrate so new records appear in priming on the next
        # user prompt without manual ``prime daemon restart``. Lazy
        # import keeps non-sleep CLI paths (``prime coldstart``, ``init``,
        # etc.) from transitively importing daemon code. Sleep cycle exit
        # code stays 0 regardless of reload outcome — records are durably
        # on disk and the next daemon start picks them up. Spec §4.4.
        try:
            from priming_stream.daemon import client as daemon_client
            result = daemon_client.reload_daemon(timeout_s=5.0)
            if result is not None:
                print(
                    f"[sleep-finalize] daemon reloaded: "
                    f"{result.get('records_before', '?')} -> "
                    f"{result.get('records_after', '?')} records "
                    f"in {result.get('reload_ms', 0):.0f}ms"
                )
            # else: daemon not running, silent skip
        except Exception as exc:  # noqa: BLE001 - non-fatal per spec §5 #6
            print(f"[sleep-finalize] daemon reload failed: {exc}", file=sys.stderr)

        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"sleep-finalize failed: {exc}", file=sys.stderr)
        return 1
