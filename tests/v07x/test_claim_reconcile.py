"""Claim reconcile — near-clone collapse + cross-session supersedence mechanics.

Deterministic parts of the claim dedup/supersedence pass (the claim sibling of
``test_card_reconcile``): the embedding candidate filter, the judge-pair plan,
the verdict→deletion resolution (with the hallucinated-id guard), and the apply
that soft-deletes an existing node vs trashes a this-cycle STAGED row
(SQL-canonical: both land in the ``records_trash`` table, reversibly).
"""
from __future__ import annotations

from priming_stream.core.db import connect
from priming_stream.core.graph_repo import GraphRepo
from priming_stream.core.models import Record, now_iso
from priming_stream.core.schema import apply_migrations
from priming_stream.ingest.claim_reconcile import (
    CLAIM_CANDIDATE_THRESHOLD,
    ClaimCandidate,
    apply_claim_deletes,
    find_claim_candidates,
    plan_claim_merges,
    resolve_deletes,
)
from priming_stream.integrations.vec_index import VecHit


# -- fakes / helpers ------------------------------------------------------

class _FakeVec:
    """Drop-in for ``RecordsVecIndex.search`` — canned, already-ordered hits so
    the filter logic is tested without fastembed. Also records delete calls."""

    def __init__(self, hits: list[VecHit] | None = None) -> None:
        self._hits = hits or []
        self.deleted: list[str] = []

    def search(self, query_text: str, k: int) -> list[VecHit]:
        return self._hits[:k]

    def delete_record(self, rid: str) -> None:
        self.deleted.append(rid)


def _repo_with(records: list[Record]) -> GraphRepo:
    conn = connect(":memory:")
    apply_migrations(conn)
    repo = GraphRepo(conn)
    for r in records:
        repo.create_record(r)
    return repo


def _claim(rid: str, summary: str = "a claim", source_date: str | None = None) -> Record:
    return Record(
        id=rid, source_uri="qmd://x/y.md",
        anchor_offset_start=0, anchor_offset_end=1,
        summary=summary, created_at=now_iso(),
        kind="claim", source_date=source_date,
    )


def _card(rid: str, key: str) -> Record:
    return Record(
        id=rid, source_uri=f"doc://{key}",
        anchor_offset_start=0, anchor_offset_end=0,
        summary=f"body of {key}", created_at=now_iso(),
        kind="index_card", doc_key=key, provisional=True,
    )


def _incoming(rid: str, summary: str = "incoming claim", source_date: str = "2026-06-10T00:00:00Z") -> Record:
    """A this-cycle staged claim (Record shape)."""
    return Record(
        id=rid, source_uri="qmd://x/y.md",
        anchor_offset_start=0, anchor_offset_end=1,
        summary=summary, created_at=now_iso(),
        kind="claim", source_date=source_date,
    )


# -- find_claim_candidates -----------------------------------------------

def test_find_candidates_filters_to_claims_above_threshold():
    repo = _repo_with([
        _claim("rec_old"),
        _card("rec_card", "t:deyne"),  # a card with high score must be dropped
    ])
    vec = _FakeVec([
        VecHit("rec_card", 0.95, "body"),  # highest but a card -> dropped
        VecHit("rec_old", 0.82, "body"),   # claim, above threshold -> kept
        VecHit("rec_low", 0.40, "body"),   # below threshold -> dropped
    ])
    out = find_claim_candidates("x", vec_index=vec, repo=repo, threshold=0.6)
    assert [c.record.id for c in out] == ["rec_old"]
    assert isinstance(out[0], ClaimCandidate)
    assert out[0].score == 0.82


def test_find_candidates_excludes_self_id():
    repo = _repo_with([_claim("rec_self")])
    vec = _FakeVec([VecHit("rec_self", 0.99, "body")])
    out = find_claim_candidates(
        "x", vec_index=vec, repo=repo, threshold=0.6, exclude_id="rec_self")
    assert out == []


def test_find_candidates_orders_desc_and_caps_k():
    repo = _repo_with([_claim("rec_a"), _claim("rec_b"), _claim("rec_c")])
    vec = _FakeVec([
        VecHit("rec_b", 0.70, "b"),
        VecHit("rec_a", 0.90, "a"),
        VecHit("rec_c", 0.80, "c"),
    ])
    out = find_claim_candidates("x", vec_index=vec, repo=repo, threshold=0.6, k=2)
    assert [c.record.id for c in out] == ["rec_a", "rec_c"]


def test_find_candidates_empty_body_returns_empty():
    repo = _repo_with([_claim("rec_a")])
    vec = _FakeVec([VecHit("rec_a", 0.99, "a")])
    assert find_claim_candidates("  ", vec_index=vec, repo=repo, threshold=0.6) == []


def test_claim_candidate_threshold_recall_first():
    assert 0.0 < CLAIM_CANDIDATE_THRESHOLD < 0.80


# -- plan_claim_merges ----------------------------------------------------

def test_plan_emits_judge_pair_with_ids_and_dates():
    repo = _repo_with([_claim("rec_old", "old claim", source_date="2026-05-01T00:00:00Z")])
    vec = _FakeVec([VecHit("rec_old", 0.81, "old claim")])
    plan = plan_claim_merges(
        [_incoming("rec_new", "new claim", source_date="2026-06-10T00:00:00Z")],
        repo=repo, vec_index=vec, threshold=0.6)
    assert len(plan["judge_pairs"]) == 1
    jp = plan["judge_pairs"][0]
    assert jp["pair_id"] == 0
    assert jp["incoming_id"] == "rec_new" and jp["survivor_id"] == "rec_old"
    assert jp["incoming_body"] == "new claim" and jp["survivor_body"] == "old claim"
    assert jp["incoming_date"] == "2026-06-10T00:00:00Z"
    assert jp["survivor_date"] == "2026-05-01T00:00:00Z"
    assert jp["score"] == 0.81


def test_plan_no_candidate_emits_nothing():
    repo = _repo_with([_claim("rec_old")])
    vec = _FakeVec([VecHit("rec_old", 0.30, "old")])  # below threshold
    plan = plan_claim_merges([_incoming("rec_new")], repo=repo, vec_index=vec, threshold=0.6)
    assert plan == {"judge_pairs": []}


def test_plan_skips_incoming_without_body():
    repo = _repo_with([_claim("rec_old")])
    vec = _FakeVec([VecHit("rec_old", 0.99, "old")])
    plan = plan_claim_merges([_incoming("rec_new", summary="   ")], repo=repo, vec_index=vec, threshold=0.6)
    assert plan == {"judge_pairs": []}


# -- resolve_deletes ------------------------------------------------------

def _pair(pid, inc, surv):
    return {"pair_id": pid, "incoming_id": inc, "incoming_body": "i",
            "survivor_id": surv, "survivor_body": "s"}


def test_resolve_takes_contradiction_and_near_clone():
    plan = {"judge_pairs": [_pair(0, "rec_n", "rec_o"), _pair(1, "rec_n2", "rec_o2")]}
    verdicts = {
        0: {"verdict": "contradiction", "delete_id": "rec_o"},
        1: {"verdict": "near-clone", "delete_id": "rec_n2"},
    }
    out = resolve_deletes(plan, verdicts)
    ids = {d["delete_id"]: d["verdict"] for d in out}
    assert ids == {"rec_o": "contradiction", "rec_n2": "near-clone"}


def test_resolve_distinct_is_noop():
    plan = {"judge_pairs": [_pair(0, "rec_n", "rec_o")]}
    out = resolve_deletes(plan, {0: {"verdict": "distinct", "delete_id": None}})
    assert out == []


def test_resolve_guards_hallucinated_id():
    # delete_id is neither member of the pair -> dropped (no deletion)
    plan = {"judge_pairs": [_pair(0, "rec_n", "rec_o")]}
    out = resolve_deletes(plan, {0: {"verdict": "contradiction", "delete_id": "rec_ELSEWHERE"}})
    assert out == []


def test_resolve_guards_null_delete_id():
    plan = {"judge_pairs": [_pair(0, "rec_n", "rec_o")]}
    out = resolve_deletes(plan, {0: {"verdict": "contradiction", "delete_id": None}})
    assert out == []


def test_resolve_dedupes_same_id():
    plan = {"judge_pairs": [_pair(0, "rec_n", "rec_o"), _pair(1, "rec_n2", "rec_o")]}
    verdicts = {
        0: {"verdict": "contradiction", "delete_id": "rec_o"},
        1: {"verdict": "near-clone", "delete_id": "rec_o"},
    }
    out = resolve_deletes(plan, verdicts)
    assert [d["delete_id"] for d in out] == ["rec_o"]  # once


def test_resolve_missing_verdict_is_noop():
    plan = {"judge_pairs": [_pair(0, "rec_n", "rec_o")]}
    assert resolve_deletes(plan, {}) == []


# -- apply_claim_deletes --------------------------------------------------

def test_apply_deletes_existing_node():
    repo = _repo_with([_claim("rec_old", "old false claim")])
    vec = _FakeVec()

    metrics = apply_claim_deletes(
        [{"delete_id": "rec_old", "verdict": "contradiction"}],
        repo=repo, vec_index=vec)

    assert metrics["deleted_existing"] == 1 and metrics["contradiction"] == 1
    assert repo.get_record("rec_old") is None        # dropped from records
    assert vec.deleted == ["rec_old"]                # dropped from vec
    trashed = repo.get_trashed("rec_old")            # ...reversibly
    assert trashed is not None
    assert trashed.summary == "old false claim"


def test_apply_trashes_this_cycle_claim():
    repo = _repo_with([])  # rec_new is NOT in the substrate yet (this cycle)
    repo.stage_record(_incoming("rec_new", "new redundant claim"))
    vec = _FakeVec()

    metrics = apply_claim_deletes(
        [{"delete_id": "rec_new", "verdict": "near-clone"}],
        repo=repo, vec_index=vec)

    assert metrics["deleted_incoming"] == 1 and metrics["near_clone"] == 1
    assert metrics["deleted_existing"] == 0
    assert vec.deleted == []                          # nothing in vec yet
    assert repo.get_staged("rec_new") is None         # staged row gone
    assert repo.get_trashed("rec_new") is not None    # ...into trash


def test_apply_skips_when_id_unknown():
    repo = _repo_with([])  # not existing, not staged
    metrics = apply_claim_deletes(
        [{"delete_id": "rec_ghost", "verdict": "near-clone"}],
        repo=repo, vec_index=None)
    assert metrics["skipped"] == 1
    assert metrics["deleted_incoming"] == 0


def test_apply_empty_is_noop():
    repo = _repo_with([])
    metrics = apply_claim_deletes([], repo=repo, vec_index=None)
    assert metrics == {"near_clone": 0, "contradiction": 0,
                       "deleted_existing": 0, "deleted_incoming": 0, "skipped": 0}


def test_trash_then_restore_round_trip():
    """The soft delete is reversible: restore moves the row back intact."""
    repo = _repo_with([_claim("rec_old", "a perfectly valid claim",
                              source_date="2026-05-01T00:00:00Z")])
    repo.trash_record("rec_old", reason="reconcile-contradiction")
    assert repo.get_record("rec_old") is None
    restored = repo.restore_record("rec_old")
    assert restored is not None
    live = repo.get_record("rec_old")
    assert live.summary == "a perfectly valid claim"
    assert live.source_date == "2026-05-01T00:00:00Z"
    assert repo.get_trashed("rec_old") is None
