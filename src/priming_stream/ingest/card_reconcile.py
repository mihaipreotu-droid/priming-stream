"""Deterministic mechanics for index-card reconciliation (piece3-B/C dedup).

The dedup DECISION ("is this the same document?") is made by an exact
``content_hash`` match (certain) or an LLM judge over embedding-similar
candidates (the Workflow). This module holds the deterministic parts around
that decision — the cheap candidate FILTER that feeds the judge, plus the
mechanics that APPLY a decided merge:

- ``find_card_candidates`` — vec-search the substrate for existing index_cards
  similar to an incoming card's body. The cheap embedding filter; its output is
  the (small) set of cards the LLM judge reads to decide same-document. Pure
  given an injected vec index + repo.
- ``merge_card_record`` — given the SURVIVING card (the existing substrate
  node, which keeps its identity) and an INCOMING card (a staged row this
  cycle), produce the merged Record: survivor's id / doc_key / created_at,
  with the FULLER content winning (a real file-card upgrades a provisional
  conversation stub).
- ``apply_card_merges`` — write the decided merges: ``replace_record`` the
  survivor with the merged content (+ re-embed), re-point this-cycle staged
  claims off the absorbed key, drop the absorbed staged card.

SQL-canonical (2026-06-12): incoming cards are STAGED rows
(``records_staging``), not ``.md`` files — the plan carries staging ids, the
apply mutates rows. Survivor = the EXISTING node by design, so existing
claims keep their links and only staged claims need re-pointing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from priming_stream.core.models import Record

if TYPE_CHECKING:  # avoid a runtime import cycle / heavy deps on this module
    from priming_stream.core.graph_repo import GraphRepo
    from priming_stream.integrations.vec_index import RecordsVecIndex


# Calibrated candidate cutoff (tuned against a real substrate).
# Same-document pairs (incoming worded
# differently from the stub) scored 0.80-0.95; different-document near-domain
# papers (the word-association cluster) topped out at 0.56; far documents < 0.26.
# 0.60 sits in the clean (0.56, 0.80) gap, leaning toward recall: the filter
# must not miss a real duplicate (a false split silently defeats the reconcile),
# while over-inclusion only costs one LLM-judge call that returns NO. The judge
# is the precision gate; this is just the cheap pre-filter.
CARD_CANDIDATE_THRESHOLD = 0.60


@dataclass
class CardCandidate:
    """An existing index_card the embedding filter flagged as a possible
    duplicate of an incoming card — a (record, similarity) pair the LLM judge
    will read. ``score`` is the cosine-like vec similarity in [0, 1]."""

    record: Record
    score: float


def find_card_candidates(
    incoming_body: str,
    *,
    vec_index: "RecordsVecIndex",
    repo: "GraphRepo",
    threshold: float,
    k: int = 5,
    vec_k: int = 60,
    exclude_doc_key: str | None = None,
) -> list[CardCandidate]:
    """Existing index_cards whose body is embedding-similar to ``incoming_body``
    — the cheap candidate filter the LLM judge then reads.

    The vec index holds ALL records (claims + cards), so we over-fetch ``vec_k``
    nearest neighbours and keep only ``kind == 'index_card'`` hits at or above
    ``threshold``, excluding the incoming card's own ``doc_key`` (self, e.g. a
    re-ingest already in the index). Returns up to ``k`` candidates,
    score-descending. An empty list means no merge candidate — the plan emits a
    fresh node (conservative: a false split is safe, a false merge loses
    content).

    ``threshold`` is intentionally a required arg, not a default: the right
    cutoff is empirical (calibrated on the De Deyne stub-vs-file pair against
    near-domain papers), so callers pass the calibrated value explicitly rather
    than inheriting a guess hidden in this signature.
    """
    if not incoming_body or not incoming_body.strip():
        return []
    hits = vec_index.search(incoming_body, vec_k)
    out: list[CardCandidate] = []
    for hit in hits:
        if hit.score < threshold:
            continue
        rec = repo.get_record(hit.record_id)
        if rec is None or rec.kind != "index_card":
            continue
        if exclude_doc_key is not None and rec.doc_key == exclude_doc_key:
            continue
        out.append(CardCandidate(record=rec, score=hit.score))
    out.sort(key=lambda c: c.score, reverse=True)
    return out[:k]


def incoming_is_fuller(survivor: Record, incoming: Record) -> bool:
    """A non-provisional (file-backed) card upgrades a provisional stub. Any
    other combination keeps the survivor's content (it was here first)."""
    return (not incoming.provisional) and bool(survivor.provisional)


def merge_card_record(survivor: Record, incoming: Record) -> Record:
    """The merged card: survivor's identity, fuller content wins.

    ``survivor`` is the existing node (keeps id / doc_key / created_at).
    ``incoming`` is this cycle's staged card. If the incoming card is fuller
    (a file-card over a stub) its body / source / content_hash / title /
    source_uri / provisional replace the survivor's; otherwise the survivor's
    content is kept.
    """
    if incoming_is_fuller(survivor, incoming):
        body = incoming.summary
        source = incoming.source
        content_hash = incoming.content_hash
        title = incoming.title
        source_uri = incoming.source_uri
        provisional = False
    else:
        body = survivor.summary
        source = survivor.source
        content_hash = survivor.content_hash
        title = survivor.title
        source_uri = survivor.source_uri
        provisional = survivor.provisional
    return Record(
        id=survivor.id,
        source_uri=source_uri,
        anchor_offset_start=survivor.anchor_offset_start,
        anchor_offset_end=survivor.anchor_offset_end,
        summary=body,
        created_at=survivor.created_at,
        kind="index_card",
        doc_key=survivor.doc_key,
        source=source,
        content_hash=content_hash,
        title=title,
        provisional=provisional,
        source_date=survivor.source_date,
    )


# -- the plan / resolve / apply pipeline ---------------------------------
#
# These functions are the reconcile orchestration the CLI
# (``prime reconcile-plan`` / ``reconcile-apply``) wraps with file I/O. They
# take injected ``repo`` + ``vec_index`` so they unit-test with fakes (no
# fastembed). The DECISION on rule-2 pairs is made by the LLM judge Workflow
# between plan and apply; these functions only build the candidate set and
# apply already-decided merges.


def plan_card_merges(
    incoming_cards: list[Record],
    *,
    repo: "GraphRepo",
    vec_index: "RecordsVecIndex",
    threshold: float = CARD_CANDIDATE_THRESHOLD,
) -> dict:
    """Build the reconcile plan for this cycle's freshly-staged cards.

    ``incoming_cards`` is the staged index_card rows (``repo.list_staged``).
    For each card whose ``doc_key`` is NOT yet an existing substrate card (a
    genuinely new key — the cross-key duplicate risk; a same-key card is left
    to ``sleep-finalize``'s upsert):

    - **Rule 1 — exact content_hash** → an existing card with identical source
      bytes is the same document with certainty. Emitted as an ``auto_merge``
      (no judge).
    - **Rule 2 — embedding candidates** → ``find_card_candidates`` surfaces
      existing cards similar enough to *maybe* be the same document; each
      becomes a ``judge_pair`` the LLM Workflow rules on. None → no merge (the
      card stays and finalize creates a fresh node; a false split is safe).

    Returns ``{"auto_merges": [...], "judge_pairs": [...]}`` — JSON-serializable
    dicts. ``judge_pairs`` carry both cards' title+body so the judge needs no DB
    access; they are ordered by descending similarity within each incoming, and
    globally ``pair_id``-numbered so ``apply`` can pick the first confident YES.
    """
    auto_merges: list[dict] = []
    judge_pairs: list[dict] = []
    pair_id = 0
    for card in incoming_cards:
        doc_key = card.doc_key
        if not doc_key:
            continue  # a keyless card is malformed; finalize will trash it
        if repo.get_record_by_doc_key(doc_key) is not None:
            continue  # same-key card — finalize's create/replace path owns it

        # Rule 1: identical source bytes -> same document, certain, no judge.
        chash = card.content_hash
        survivor = repo.get_card_by_content_hash(chash) if chash else None
        if survivor is not None and survivor.doc_key != doc_key:
            auto_merges.append({
                "incoming_id": card.id,
                "incoming_doc_key": doc_key,
                "survivor_id": survivor.id,
                "survivor_doc_key": survivor.doc_key,
                "via": "content_hash",
            })
            continue

        # Rule 2: embedding candidates -> judge pairs (conservative; the judge
        # decides). exclude_doc_key guards against a self-match on re-runs.
        cands = find_card_candidates(
            card.summary, vec_index=vec_index, repo=repo,
            threshold=threshold, exclude_doc_key=doc_key,
        )
        for cand in cands:
            judge_pairs.append({
                "pair_id": pair_id,
                "incoming_id": card.id,
                "incoming_doc_key": doc_key,
                "incoming_title": card.title or "",
                "incoming_body": card.summary,
                "survivor_id": cand.record.id,
                "survivor_doc_key": cand.record.doc_key,
                "survivor_title": cand.record.title or "",
                "survivor_body": cand.record.summary,
                "score": round(cand.score, 4),
            })
            pair_id += 1
    return {"auto_merges": auto_merges, "judge_pairs": judge_pairs}


def resolve_merges(plan: dict, verdicts: dict[int, bool]) -> list[dict]:
    """Collapse a plan + the judge's verdicts into the merges to apply.

    ``verdicts`` maps ``pair_id -> same?`` (a missing pair_id is treated as NO —
    conservative). Every ``auto_merge`` is taken; for ``judge_pairs``, each
    incoming card merges into the FIRST candidate the judge confirmed (pairs are
    ordered by descending similarity, so "first YES" = best confirmed match).

    A survivor already claimed by an earlier merge this cycle is NOT reused for
    a second incoming (that would overwrite the first merge); the later incoming
    is dropped from the merge set and stays a fresh node — surfaced via the
    returned ``skipped_survivor_conflict`` is the caller's to log. Returns the
    ordered merge list; each merge = ``{incoming_id, incoming_doc_key,
    survivor_id, survivor_doc_key, via}``.
    """
    merges: list[dict] = []
    used_survivors: set[str] = set()
    used_incoming: set[str] = set()

    def _take(m: dict) -> None:
        sid = m["survivor_id"]
        inc = m["incoming_doc_key"]
        if inc in used_incoming or sid in used_survivors:
            return
        used_incoming.add(inc)
        used_survivors.add(sid)
        merges.append(m)

    for m in plan.get("auto_merges", []):
        _take({**m, "via": m.get("via", "content_hash")})

    # judge_pairs are in pair_id order = descending score within each incoming.
    for p in plan.get("judge_pairs", []):
        if p["incoming_doc_key"] in used_incoming:
            continue  # this incoming already merged (auto or earlier YES)
        if not verdicts.get(p["pair_id"], False):
            continue
        _take({
            "incoming_id": p["incoming_id"],
            "incoming_doc_key": p["incoming_doc_key"],
            "survivor_id": p["survivor_id"],
            "survivor_doc_key": p["survivor_doc_key"],
            "via": "judge",
        })
    return merges


def apply_card_merges(
    merges: list[dict],
    *,
    repo: "GraphRepo",
    vec_index: "RecordsVecIndex | None" = None,
) -> dict:
    """Apply resolved merges to the substrate (the deterministic write).

    For each merge: ``replace_record`` the SURVIVOR with the merged content
    (survivor identity, fuller content wins) and re-embed it when its body
    changed; re-point this-cycle staged claims off the absorbed key onto the
    survivor's key; drop the absorbed staged card so finalize never promotes
    a duplicate node. Survivor + incoming are re-read from SQLite here
    (truth), not trusted from the plan. The vec write is best-effort
    (rebuildable). Returns counts.
    """
    metrics = {
        "merged": 0, "auto": 0, "judged": 0,
        "claims_repointed": 0, "skipped": 0,
        # survivor ids whose body changed but could NOT be re-embedded (vec
        # index absent/failed) — an explicit run-vec-index-rebuild signal,
        # not a silent stale embedding.
        "vec_stale": [],
    }
    for m in merges:
        survivor = repo.get_record(m["survivor_id"])
        incoming = repo.get_staged(m["incoming_id"])
        if survivor is None or incoming is None:
            metrics["skipped"] += 1
            continue

        merged = merge_card_record(survivor, incoming)
        body_changed = merged.summary != survivor.summary
        repo.replace_record(survivor.id, merged)
        if body_changed:
            embedded = False
            if vec_index is not None:
                try:
                    vec_index.add_record(merged.id, merged.summary)
                    embedded = True
                except Exception:  # noqa: BLE001 - vec is rebuildable
                    pass
            if not embedded:
                metrics["vec_stale"].append(merged.id)

        metrics["claims_repointed"] += repo.repoint_staged_doc_key(
            m["incoming_doc_key"], m["survivor_doc_key"],
        )
        repo.delete_staged(incoming.id)

        metrics["merged"] += 1
        metrics["auto" if m.get("via") == "content_hash" else "judged"] += 1
    return metrics
