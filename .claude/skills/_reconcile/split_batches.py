"""Split the reconcile judge plans into small per-batch input files.

WHY: the judge Workflow's JS sandbox cannot read files, so it used to delegate
"load the whole plan" to ONE agent that read the entire plan and re-emitted every
pair verbatim. That balloons context and HANGS once a plan is large (a coldstart
``--limit 40`` cycle produced 323 claim pairs / ~250 KB → the load agent paged the
file with repeated Reads, then had to regenerate ~250 KB of structured output, and
froze). The judge *batch* agents were never the problem — they get their pairs
inline. Only the monolithic load step did not scale.

Fix: do the reading + pooling + batching in Python (no LLM, no hang). Each judge
agent then reads ONLY its own small batch file (≤ BATCH pooled pairs ≈ a few tens
of KB). Card + claim pairs are pooled and batched together, exactly as the old
in-JS pooling did, so verdict semantics are unchanged.

Outputs (under ``storage/corpus/_judge_batches/``):
  - ``batch_<i>.json``  = {batch_index, card_verdicts_dir, claim_verdicts_dir, items:[...]}
  - ``manifest.json``   = {n_batches, batch_dir, card_verdicts_dir, claim_verdicts_dir}

The workflow reads the tiny manifest via one agent, then fans out one agent per
batch file. Verdict output format + dirs are identical to before, so the existing
``reconcile-apply`` / ``claim-reconcile-apply`` readers are untouched.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

# Repo-root-relative: this file lives at <repo>/.claude/skills/_reconcile/, so
# parents[3] is the repo root. storage/corpus is created there by `prime init`.
BASE = Path(__file__).resolve().parents[3] / "storage" / "corpus"
CARD_PLAN = BASE / "_reconcile_plan.json"
CLAIM_PLAN = BASE / "_claim_reconcile_plan.json"
BATCH_DIR = BASE / "_judge_batches"
BATCH = 40

# only the fields the judge actually needs (drop score etc. to keep batches lean)
_CARD_FIELDS = ("pair_id", "incoming_title", "incoming_body",
                "survivor_title", "survivor_body")
_CLAIM_FIELDS = ("pair_id", "incoming_id", "incoming_body", "incoming_date",
                 "survivor_id", "survivor_body", "survivor_date")


def _load(path: Path) -> dict:
    if not path.exists():
        return {"judge_pairs": [], "verdicts_dir": ""}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"judge_pairs": [], "verdicts_dir": ""}


def _pick(pair: dict, fields: tuple[str, ...], kind: str) -> dict:
    out = {"kind": kind}
    for f in fields:
        if f in pair:
            out[f] = pair[f]
    return out


def main() -> int:
    card = _load(CARD_PLAN)
    claim = _load(CLAIM_PLAN)
    card_dir = card.get("verdicts_dir") or ""
    claim_dir = claim.get("verdicts_dir") or ""

    items: list[dict] = []
    for p in card.get("judge_pairs", []):
        items.append(_pick(p, _CARD_FIELDS, "card"))
    for p in claim.get("judge_pairs", []):
        items.append(_pick(p, _CLAIM_FIELDS, "claim"))

    # fresh batch dir each run (scratch; regenerable)
    if BATCH_DIR.exists():
        shutil.rmtree(BATCH_DIR)
    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    batches = [items[i:i + BATCH] for i in range(0, len(items), BATCH)]
    for i, batch in enumerate(batches):
        (BATCH_DIR / f"batch_{i}.json").write_text(
            json.dumps({
                "batch_index": i,
                "card_verdicts_dir": card_dir,
                "claim_verdicts_dir": claim_dir,
                "items": batch,
            }, ensure_ascii=False),
            encoding="utf-8",
        )

    manifest = {
        "n_batches": len(batches),
        "batch_dir": str(BATCH_DIR),
        "card_verdicts_dir": card_dir,
        "claim_verdicts_dir": claim_dir,
    }
    (BATCH_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    n_card = sum(1 for x in items if x["kind"] == "card")
    n_claim = sum(1 for x in items if x["kind"] == "claim")
    print(f"split-batches: {len(items)} pair(s) ({n_card} card, {n_claim} claim) "
          f"-> {len(batches)} batch file(s) of <={BATCH} in {BATCH_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
