"""Deterministic mechanics for CLAIM reconciliation — near-clone collapse +
cross-session supersedence (the claim sibling of ``card_reconcile``).

Where ``card_reconcile`` dedups *documents* by merging into a survivor, this
module handles *claims*: a new conversation can restate (near-clone) or refute
(supersede) a claim already in the substrate. The DECISION ("same proposition?
contradiction? which is false?") is an LLM judge over embedding-similar
candidates (the Workflow). This module holds the deterministic parts around it:

- ``find_claim_candidates`` — vec-search the substrate for existing *claims*
  similar to an incoming claim. The cheap embedding filter feeding the judge.
- ``plan_claim_merges`` — build the judge-pair plan for this cycle's new claims.
- ``resolve_deletes`` — collapse plan + verdicts into the set of record ids to
  delete (guarded against a hallucinated id outside the pair).
- ``apply_claim_deletes`` — delete each: an EXISTING claim (live in
  SQLite/Chroma) is soft-deleted to the ``records_trash`` table + dropped from
  vec; a THIS-CYCLE claim (still staged, never promoted) moves staging →
  trash so finalize never promotes it.

Unlike cards there is no merge and no content_hash auto-rule: claims have no
``doc_key`` identity, every candidate pair goes to the judge, and the resolution
is a *deletion* (the false/redundant record is removed), not a content merge.

Timing: reconcile runs BEFORE ``sleep-finalize``. This cycle's new claims are
STAGED rows (``records_staging``) not yet in ``records``; prior claims are in
``records`` + ChromaDB. So candidate search hits only existing claims — exactly
the cross-session case. New-vs-new within one cycle is out of scope here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid a runtime import cycle / heavy deps on this module
    from priming_stream.core.graph_repo import GraphRepo
    from priming_stream.core.models import Record
    from priming_stream.integrations.vec_index import RecordsVecIndex


# Recall-first candidate cutoff. A refutation shares most tokens with what it
# refutes ("X holds" vs "X does not hold" embed close), so the filter must not
# miss the conflicting neighbour; over-inclusion only costs one judge call that
# returns "distinct". The judge is the precision gate.
# Calibration of the candidate cutoff:
# HELD at 0.60. The probe found anchored same-subject contradictions clear
# 0.60 comfortably (band 0.66-0.84 once incoming is anchor-faithful, not on the
# floor as the stale pre-anchor fixture suggested); 0.55 was only optional insurance
# for the unanchored / cross-anchor tail (~500 anchor-less claims) and was not
# adopted. The eviction risk was instead addressed via CLAIM_CANDIDATE_K below.
CLAIM_CANDIDATE_THRESHOLD = 0.60

# Top-k candidates per incoming claim handed to the judge. Phase-5 Module-2 raised
# 5 -> 8: in the dense substrate a true contradiction can rank *below* same-topic
# *distinct* neighbours and get evicted by a tight cap (a negation surfaced at
# rank 4-5). k=8 restores eviction margin; extra candidates only cost judge calls
# that return "distinct". This is the better-targeted recall lever than threshold.
CLAIM_CANDIDATE_K = 8

# Verdicts that trigger a deletion (vs "distinct" which is a no-op).
_DELETE_VERDICTS = ("near-clone", "contradiction")


@dataclass
class ClaimCandidate:
    """An existing claim the embedding filter flagged as a possible duplicate or
    conflict of an incoming claim — a (record, similarity) pair the LLM judge
    reads. ``score`` is the cosine-like vec similarity in [0, 1]."""

    record: "Record"
    score: float


def find_claim_candidates(
    incoming_summary: str,
    *,
    vec_index: "RecordsVecIndex",
    repo: "GraphRepo",
    threshold: float,
    exclude_id: str | None = None,
    k: int = CLAIM_CANDIDATE_K,
    vec_k: int = 60,
) -> list[ClaimCandidate]:
    """Existing claims whose summary is embedding-similar to ``incoming_summary``
    — the cheap candidate filter the LLM judge then reads.

    The vec index holds ALL records (claims + cards), so we over-fetch ``vec_k``
    nearest neighbours and keep only ``kind == 'claim'`` hits at or above
    ``threshold``, excluding ``exclude_id`` (the incoming claim's own id, a
    defensive guard — it is not in the index yet pre-finalize). Returns up to
    ``k`` candidates, score-descending. An empty list means no neighbour worth
    judging (conservative: leave the claim as a fresh node).
    """
    if not incoming_summary or not incoming_summary.strip():
        return []
    hits = vec_index.search(incoming_summary, vec_k)
    out: list[ClaimCandidate] = []
    for hit in hits:
        if hit.score < threshold:
            continue
        if exclude_id is not None and hit.record_id == exclude_id:
            continue
        rec = repo.get_record(hit.record_id)
        if rec is None or rec.kind != "claim":
            continue
        out.append(ClaimCandidate(record=rec, score=hit.score))
    out.sort(key=lambda c: c.score, reverse=True)
    return out[:k]


# -- the plan / resolve / apply pipeline ---------------------------------
#
# These functions are the reconcile orchestration the CLI
# (``prime claim-reconcile-plan`` / ``claim-reconcile-apply``) wraps with file
# I/O. They take injected ``repo`` + ``vec_index`` so they unit-test with fakes
# (no fastembed). The DECISION on each pair is the LLM judge Workflow between
# plan and apply; these functions only build candidates and apply decided deletes.


def plan_claim_merges(
    incoming_claims: "list[Record]",
    *,
    repo: "GraphRepo",
    vec_index: "RecordsVecIndex",
    threshold: float = CLAIM_CANDIDATE_THRESHOLD,
) -> dict:
    """Build the reconcile plan for this cycle's freshly-staged claims.

    ``incoming_claims`` is the staged claim rows (``repo.list_staged``, each
    kind == 'claim', not yet promoted). For each, surface embedding-similar
    existing claims; each becomes a ``judge_pair`` the LLM Workflow classifies
    as near-clone / contradiction / distinct and, when a deletion is
    warranted, names the record to delete.

    Returns ``{"judge_pairs": [...]}`` — JSON-serializable. Pairs carry both ids,
    both bodies, both ``source_date``s (so the judge can reason about which stance
    is later), and the similarity score. No DB access needed by the judge.
    """
    judge_pairs: list[dict] = []
    pair_id = 0
    for claim in incoming_claims:
        if not claim.id or not claim.summary.strip():
            continue
        cands = find_claim_candidates(
            claim.summary, vec_index=vec_index, repo=repo,
            threshold=threshold, exclude_id=claim.id,
        )
        for cand in cands:
            judge_pairs.append({
                "pair_id": pair_id,
                "incoming_id": claim.id,
                "incoming_body": claim.summary,
                "incoming_date": claim.source_date or "",
                "survivor_id": cand.record.id,
                "survivor_body": cand.record.summary,
                "survivor_date": cand.record.source_date or "",
                "score": round(cand.score, 4),
            })
            pair_id += 1
    return {"judge_pairs": judge_pairs}


def resolve_deletes(plan: dict, verdicts: dict[int, dict]) -> list[dict]:
    """Collapse a plan + the judge's verdicts into the deletions to apply.

    ``verdicts`` maps ``pair_id -> {"verdict": str, "delete_id": str | None}``.
    A deletion is taken only when the verdict is near-clone/contradiction AND the
    named ``delete_id`` is one of that pair's two records (guard against a judge
    that names an id outside the pair). Each id is deleted at most once; the
    returned dicts carry the ``verdict`` and the ``pair_id`` that triggered it for
    auditing. A missing / malformed / "distinct" verdict is a no-op (conservative).
    """
    out: list[dict] = []
    seen: set[str] = set()
    for p in plan.get("judge_pairs", []):
        pid = p["pair_id"]
        v = verdicts.get(pid)
        if not v:
            continue
        verdict = (v.get("verdict") or "").strip().lower()
        if verdict not in _DELETE_VERDICTS:
            continue
        delete_id = v.get("delete_id")
        if delete_id not in (p["incoming_id"], p["survivor_id"]):
            continue  # hallucinated / null id — skip, no deletion
        if delete_id in seen:
            continue
        seen.add(delete_id)
        out.append({"delete_id": delete_id, "verdict": verdict, "pair_id": pid})
    return out


def apply_claim_deletes(
    deletes: list[dict],
    *,
    repo: "GraphRepo",
    vec_index: "RecordsVecIndex | None" = None,
) -> dict:
    """Apply resolved deletions to the substrate (the deterministic write).

    For each ``delete_id``:
    - **existing** (``repo.get_record`` present, live in SQLite/Chroma):
      soft-delete — row moves to ``records_trash`` (with the verdict as the
      reason) and (if a vec index is given) the embedding is dropped.
    - **this-cycle** (not live — a staged row): staging → trash, so
      ``sleep-finalize`` never promotes it. Nothing in Chroma to remove yet.

    Soft delete (trash table, not hard DELETE) keeps the deletion reversible
    at zero cost (``prime record restore``), matching ``record delete``'s
    default. No daemon reload — ``sleep-finalize`` reloads at cycle end.
    Returns counts.
    """
    metrics = {
        "near_clone": 0, "contradiction": 0,
        "deleted_existing": 0, "deleted_incoming": 0, "skipped": 0,
    }

    for d in deletes:
        rid = d["delete_id"]
        verdict = d.get("verdict", "")

        if repo.get_record(rid) is not None:
            # Existing substrate node: row -> trash, drop the embedding.
            try:
                repo.trash_record(rid, reason=f"reconcile-{verdict}")
            except Exception:  # noqa: BLE001 - leave substrate untouched on failure
                metrics["skipped"] += 1
                continue
            if vec_index is not None:
                try:
                    vec_index.delete_record(rid)
                except Exception:  # noqa: BLE001 - vec is rebuildable
                    pass
            metrics["deleted_existing"] += 1
        else:
            # This-cycle claim: staged only. Trash it so finalize skips it.
            if repo.trash_staged(rid, reason=f"reconcile-{verdict}") is None:
                metrics["skipped"] += 1
                continue
            metrics["deleted_incoming"] += 1

        if verdict == "contradiction":
            metrics["contradiction"] += 1
        elif verdict == "near-clone":
            metrics["near_clone"] += 1
    return metrics
