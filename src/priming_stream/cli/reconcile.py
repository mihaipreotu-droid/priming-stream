"""``prime reconcile-plan`` / ``prime reconcile-apply`` — the card-dedup
bookends the ``/prime-ingest`` skill shells out to, between the card writer and
``sleep-finalize``.

The problem: one document can enter the substrate from two sources with
different identity info (a conversation stub keyed ``t:<slug>`` vs a later file
card keyed ``doi:…``), so a same-key upsert would DUPLICATE it. These two
commands dedup across keys:

- ``reconcile-plan`` scans this cycle's freshly-STAGED cards
  (``records_staging``), and for each card whose ``doc_key`` is genuinely new,
  decides: exact ``content_hash`` match → auto-merge (certain); else
  embedding-similar candidates → judge-pairs. It writes
  ``_reconcile_plan.json`` + resets ``_reconcile_verdicts/``. The LLM judge
  Workflow runs in between (it reads the plan, writes verdict JSONL files) —
  ``claude -p`` does not authenticate headless here, so the judge must be a
  Workflow agent.
- ``reconcile-apply`` reads the plan + verdicts, and for each confirmed merge
  rewrites the survivor row (fuller content wins, + re-embed), re-points
  this-cycle staged claims off the absorbed key, and drops the absorbed
  staged card.

Conservative by design: a false split is safe (two nodes for one doc), a false
merge is not (it loses a document's content), so "no candidate / unsure → new
node". The vec search is only the cheap pre-filter; the judge is the gate.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from priming_stream.core.config import load_config
from priming_stream.core.db import connect
from priming_stream.core.graph_repo import GraphRepo
from priming_stream.core.paths import resolve_paths
from priming_stream.core.schema import apply_migrations
from priming_stream.ingest.card_reconcile import (
    CARD_CANDIDATE_THRESHOLD,
    apply_card_merges,
    plan_card_merges,
    resolve_merges,
)
from priming_stream.ingest.claim_reconcile import (
    CLAIM_CANDIDATE_THRESHOLD,
    apply_claim_deletes,
    plan_claim_merges,
    resolve_deletes,
)

PLAN_NAME = "_reconcile_plan.json"
VERDICTS_DIRNAME = "_reconcile_verdicts"
CLAIM_PLAN_NAME = "_claim_reconcile_plan.json"
CLAIM_VERDICTS_DIRNAME = "_claim_reconcile_verdicts"


def register(subparsers) -> None:
    p_plan = subparsers.add_parser(
        "reconcile-plan",
        help="scan this cycle's staged cards; emit a dedup plan "
             "(content_hash auto-merges + embedding judge-pairs)",
    )
    p_plan.add_argument(
        "--threshold", type=float, default=CARD_CANDIDATE_THRESHOLD,
        help=f"vec candidate cutoff (default {CARD_CANDIDATE_THRESHOLD}, "
             "calibrated)",
    )
    p_plan.set_defaults(func=cmd_reconcile_plan)

    p_apply = subparsers.add_parser(
        "reconcile-apply",
        help="apply the reconcile plan + judge verdicts (merge survivors, "
             "re-point staged claims, drop absorbed staged cards)",
    )
    p_apply.set_defaults(func=cmd_reconcile_apply)

    p_cplan = subparsers.add_parser(
        "claim-reconcile-plan",
        help="scan this cycle's staged claims; emit a dedup/supersedence plan "
             "(embedding judge-pairs vs existing claims)",
    )
    p_cplan.add_argument(
        "--threshold", type=float, default=CLAIM_CANDIDATE_THRESHOLD,
        help=f"vec candidate cutoff (default {CLAIM_CANDIDATE_THRESHOLD}, "
             "recall-first; calibrated in Phase 5)",
    )
    p_cplan.set_defaults(func=cmd_claim_reconcile_plan)

    p_capply = subparsers.add_parser(
        "claim-reconcile-apply",
        help="apply the claim reconcile plan + judge verdicts (soft-delete "
             "the false/redundant record to the records_trash table)",
    )
    p_capply.set_defaults(func=cmd_claim_reconcile_apply)


# -- helpers --------------------------------------------------------------


class _NoVec:
    """Degraded vec index: rule-2 candidate search returns nothing. Used only
    when the real ChromaDB index fails to open, so reconcile still applies the
    certain content_hash auto-merges instead of failing the whole cycle."""

    def search(self, query_text: str, k: int):  # noqa: D401 - drop-in
        return []


def _open_vec(paths, cfg, *, warn: str):
    """The real vec index, or None on failure (with a stderr warning)."""
    try:
        from priming_stream.integrations.vec_index import RecordsVecIndex
        return RecordsVecIndex(paths.vec_index_dir, cfg.vec_index.model_name)
    except Exception as exc:  # noqa: BLE001 - degrade, don't fail the cycle
        print(f"  WARN vec index unavailable ({exc.__class__.__name__}); "
              f"{warn}", file=sys.stderr)
        return None


def _iter_verdict_jsonl(verdicts_dir: Path):
    """Yield parsed JSON objects from every ``batch_*.jsonl`` verdict file the
    unified judge Workflow wrote into ``verdicts_dir`` (one object per line,
    one line per pair). Unreadable files and malformed lines are skipped —
    conservative (a dropped verdict counts as no-merge / distinct downstream)."""
    if not verdicts_dir.exists():
        return
    for f in sorted(verdicts_dir.glob("*.jsonl")):
        try:
            text = f.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(obj, dict):
                yield obj


def _read_verdicts(verdicts_dir: Path, judge_pairs: list[dict]) -> dict[int, bool]:
    """Card verdicts from the unified judge's ``batch_*.jsonl`` files. A card
    entry is ``{"kind": "card", "pair_id": N, "same": bool}``. A pair with no
    entry (missing / unparseable) is absent here and counts as NO downstream —
    conservative (no merge on doubt)."""
    want = {p["pair_id"] for p in judge_pairs}
    verdicts: dict[int, bool] = {}
    for obj in _iter_verdict_jsonl(verdicts_dir):
        pid = obj.get("pair_id")
        if pid in want and ("same" in obj or obj.get("kind") == "card"):
            verdicts[pid] = bool(obj.get("same"))
    return verdicts


# -- reconcile-plan -------------------------------------------------------


def cmd_reconcile_plan(args: argparse.Namespace) -> int:
    try:
        cfg = load_config()
        paths = resolve_paths(cfg, project_root=Path.cwd())
        if not paths.graph_db.exists():
            print(f"no graph database at {paths.graph_db} — run 'prime init' first",
                  file=sys.stderr)
            return 1

        corpus = paths.corpus_dir

        conn = connect(paths.graph_db)
        try:
            apply_migrations(conn)
            repo = GraphRepo(conn)
            incoming = repo.list_staged(kind="index_card")

            # vec index for rule-2 candidate search; degrade to rule-1-only if
            # it cannot open (the content_hash auto-merges are still certain).
            vec = _open_vec(
                paths, cfg, warn="rule-2 (embedding) candidates skipped",
            ) or _NoVec()
            plan = plan_card_merges(
                incoming, repo=repo, vec_index=vec, threshold=args.threshold,
            )
        finally:
            conn.close()

        verdicts_dir = corpus / VERDICTS_DIRNAME
        if verdicts_dir.exists():
            shutil.rmtree(verdicts_dir)
        verdicts_dir.mkdir(parents=True, exist_ok=True)

        plan["verdicts_dir"] = str(verdicts_dir)
        plan_path = corpus / PLAN_NAME
        plan_path.write_text(
            json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8",
        )

        n_auto = len(plan["auto_merges"])
        n_pairs = len(plan["judge_pairs"])
        n_incoming = len(incoming)
        # incoming cards with neither an auto-merge nor any judge-pair stay fresh
        touched = {m["incoming_doc_key"] for m in plan["auto_merges"]}
        touched |= {p["incoming_doc_key"] for p in plan["judge_pairs"]}
        # only newly-keyed incoming are reconcile candidates; the rest are
        # same-key (finalize owns them) — report the candidate set, not all.
        print(f"plan={plan_path}")
        print(f"reconcile-plan: {n_incoming} staged card(s) scanned, "
              f"{n_auto} auto-merge(s), {n_pairs} judge-pair(s) "
              f"over {len(touched)} candidate doc(s)")
        if n_pairs:
            print(f"  judge needed: run the reconcile-judge Workflow "
                  f"(writes {verdicts_dir}\\batch_*.jsonl), then reconcile-apply")
        for p in plan["judge_pairs"]:
            print(f"  pair {p['pair_id']}: score={p['score']:.3f}  "
                  f"{p['incoming_doc_key'][:34]} ?= {p['survivor_doc_key'][:34]}")
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"reconcile-plan failed: {exc}", file=sys.stderr)
        return 1


# -- reconcile-apply ------------------------------------------------------


def cmd_reconcile_apply(args: argparse.Namespace) -> int:
    try:
        cfg = load_config()
        paths = resolve_paths(cfg, project_root=Path.cwd())
        if not paths.graph_db.exists():
            print(f"no graph database at {paths.graph_db} — run 'prime init' first",
                  file=sys.stderr)
            return 1

        corpus = paths.corpus_dir
        plan_path = corpus / PLAN_NAME
        if not plan_path.exists():
            print(f"no {PLAN_NAME} — run reconcile-plan first", file=sys.stderr)
            return 1
        plan = json.loads(plan_path.read_text(encoding="utf-8"))

        verdicts_dir = Path(plan.get("verdicts_dir") or (corpus / VERDICTS_DIRNAME))
        verdicts = _read_verdicts(verdicts_dir, plan.get("judge_pairs", []))
        merges = resolve_merges(plan, verdicts)

        # vec for the survivor re-embed on a fuller-content merge; best-effort.
        vec = _open_vec(
            paths, cfg, warn="survivor re-embeds skipped (rebuildable)",
        )

        conn = connect(paths.graph_db)
        try:
            apply_migrations(conn)
            repo = GraphRepo(conn)
            metrics = apply_card_merges(merges, repo=repo, vec_index=vec)
        finally:
            conn.close()

        n_yes = sum(1 for v in verdicts.values() if v)
        print(json.dumps({"verdicts_yes": n_yes, "verdicts_total": len(verdicts),
                          "metrics": metrics}, ensure_ascii=False))
        print(f"reconcile-apply: {metrics['merged']} merge(s) applied "
              f"({metrics['auto']} auto, {metrics['judged']} judged-yes), "
              f"{metrics['claims_repointed']} claim(s) re-pointed, "
              f"{metrics['skipped']} skipped")
        if metrics.get("vec_stale"):
            print(f"  WARN stale embedding(s) for {metrics['vec_stale']} — "
                  f"run 'prime vec-index-rebuild'", file=sys.stderr)
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"reconcile-apply failed: {exc}", file=sys.stderr)
        return 1


# -- claim reconcile (near-clone + supersedence) --------------------------


def _read_claim_verdicts(
    verdicts_dir: Path, judge_pairs: list[dict],
) -> dict[int, dict]:
    """Claim verdicts from the unified judge's ``batch_*.jsonl`` files. A claim
    entry is ``{"kind": "claim", "pair_id": N, "verdict": "...",
    "delete_id": "<id>"|null}``. A pair with no entry is absent here and is a
    no-op downstream — conservative (no deletion on doubt)."""
    want = {p["pair_id"] for p in judge_pairs}
    verdicts: dict[int, dict] = {}
    for obj in _iter_verdict_jsonl(verdicts_dir):
        pid = obj.get("pair_id")
        if pid not in want or not ("verdict" in obj or obj.get("kind") == "claim"):
            continue
        delete_id = obj.get("delete_id")
        if isinstance(delete_id, str) and delete_id.strip().lower() in ("none", "", "null"):
            delete_id = None
        verdicts[pid] = {
            "verdict": (obj.get("verdict") or "distinct").strip().lower(),
            "delete_id": delete_id,
        }
    return verdicts


def cmd_claim_reconcile_plan(args: argparse.Namespace) -> int:
    try:
        cfg = load_config()
        paths = resolve_paths(cfg, project_root=Path.cwd())
        if not paths.graph_db.exists():
            print(f"no graph database at {paths.graph_db} — run 'prime init' first",
                  file=sys.stderr)
            return 1

        corpus = paths.corpus_dir

        conn = connect(paths.graph_db)
        try:
            apply_migrations(conn)
            repo = GraphRepo(conn)
            incoming = repo.list_staged(kind="claim")
            if not incoming:
                # No new claims this cycle (e.g. a document-only cycle): write an
                # empty plan and skip opening the vec index (fastembed load).
                plan = {"judge_pairs": []}
            else:
                vec = _open_vec(
                    paths, cfg,
                    warn="claim reconcile skipped (no candidates)",
                ) or _NoVec()
                plan = plan_claim_merges(
                    incoming, repo=repo, vec_index=vec, threshold=args.threshold,
                )
        finally:
            conn.close()

        verdicts_dir = corpus / CLAIM_VERDICTS_DIRNAME
        if verdicts_dir.exists():
            shutil.rmtree(verdicts_dir)
        verdicts_dir.mkdir(parents=True, exist_ok=True)

        plan["verdicts_dir"] = str(verdicts_dir)
        plan_path = corpus / CLAIM_PLAN_NAME
        plan_path.write_text(
            json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8",
        )

        n_pairs = len(plan["judge_pairs"])
        print(f"plan={plan_path}")
        print(f"claim-reconcile-plan: {len(incoming)} staged claim(s) scanned, "
              f"{n_pairs} judge-pair(s)")
        if n_pairs:
            print(f"  judge needed: run the reconcile-judge Workflow "
                  f"(writes {verdicts_dir}\\batch_*.jsonl), then "
                  f"claim-reconcile-apply")
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"claim-reconcile-plan failed: {exc}", file=sys.stderr)
        return 1


def cmd_claim_reconcile_apply(args: argparse.Namespace) -> int:
    try:
        cfg = load_config()
        paths = resolve_paths(cfg, project_root=Path.cwd())
        if not paths.graph_db.exists():
            print(f"no graph database at {paths.graph_db} — run 'prime init' first",
                  file=sys.stderr)
            return 1

        corpus = paths.corpus_dir
        plan_path = corpus / CLAIM_PLAN_NAME
        if not plan_path.exists():
            print(f"no {CLAIM_PLAN_NAME} — run claim-reconcile-plan first",
                  file=sys.stderr)
            return 1
        plan = json.loads(plan_path.read_text(encoding="utf-8"))

        verdicts_dir = Path(
            plan.get("verdicts_dir") or (corpus / CLAIM_VERDICTS_DIRNAME))
        verdicts = _read_claim_verdicts(verdicts_dir, plan.get("judge_pairs", []))
        deletes = resolve_deletes(plan, verdicts)

        vec = _open_vec(
            paths, cfg,
            warn="existing-record deletes skip the embedding (rebuildable)",
        )

        conn = connect(paths.graph_db)
        try:
            apply_migrations(conn)
            repo = GraphRepo(conn)
            metrics = apply_claim_deletes(deletes, repo=repo, vec_index=vec)
        finally:
            conn.close()

        print(json.dumps({"metrics": metrics}, ensure_ascii=False))
        print(f"claim-reconcile-apply: "
              f"{metrics['deleted_existing']} existing + "
              f"{metrics['deleted_incoming']} this-cycle record(s) deleted "
              f"({metrics['contradiction']} contradiction, "
              f"{metrics['near_clone']} near-clone), "
              f"{metrics['skipped']} skipped")
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"claim-reconcile-apply failed: {exc}", file=sys.stderr)
        return 1
