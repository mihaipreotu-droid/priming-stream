"""piece3-B/C: deterministic card-reconcile mechanics (merge + re-point).

SQL-canonical: incoming cards are STAGED rows (``records_staging``), the
merge mutates the survivor row, re-pointing is an UPDATE on staged claims.
"""
from __future__ import annotations

from priming_stream.core.db import connect
from priming_stream.core.graph_repo import GraphRepo
from priming_stream.core.models import Record, now_iso
from priming_stream.core.schema import apply_migrations
from priming_stream.ingest.card_reconcile import (
    CARD_CANDIDATE_THRESHOLD,
    CardCandidate,
    apply_card_merges,
    find_card_candidates,
    incoming_is_fuller,
    merge_card_record,
    plan_card_merges,
    resolve_merges,
)
from priming_stream.integrations.vec_index import VecHit


def _stub(**kw) -> Record:
    d = dict(
        id="rec_stub0001",
        source_uri="doc://t:deyne-2019-swow-en",
        anchor_offset_start=0, anchor_offset_end=0,
        summary="De Deyne et al. 2019 SWOW-EN, a continued word association dataset. [unverified]",
        created_at="2026-05-01T00:00:00Z",
        kind="index_card", doc_key="t:deyne-2019-swow-en",
        source=None, content_hash=None,
        title="De Deyne et al. 2019 SWOW-EN", provisional=True,
    )
    d.update(kw)
    return Record(**d)


def _incoming_full(**kw) -> Record:
    d = dict(
        id="rec_full9999",
        source_uri="file:///C:/Vault/raw/papers/deyne2019.pdf",
        anchor_offset_start=0, anchor_offset_end=0,
        summary="## Summary\nThe Small World of Words English association norms.\n\n## Key points\n- 12k cues",
        created_at="2026-06-01T00:00:00Z",
        kind="index_card",
        doc_key="doi:10.3758/s13428-018-1115-7",
        source="file:///C:/Vault/raw/papers/deyne2019.pdf",
        content_hash="abc123",
        title="The Small World of Words English word association norms",
        provisional=False,
    )
    d.update(kw)
    return Record(**d)


# -- incoming_is_fuller ---------------------------------------------------

def test_full_upgrades_stub():
    assert incoming_is_fuller(_stub(), _incoming_full()) is True


def test_stub_does_not_upgrade_full():
    survivor = _stub(provisional=False, content_hash="x")
    assert incoming_is_fuller(survivor, _incoming_full(provisional=True)) is False


def test_two_stubs_keep_survivor():
    assert incoming_is_fuller(_stub(), _incoming_full(provisional=True)) is False


# -- merge_card_record (the De Deyne upgrade) ------------------------------

def test_merge_keeps_survivor_identity_takes_fuller_content():
    merged = merge_card_record(_stub(), _incoming_full())
    # survivor identity preserved (records keep their links):
    assert merged.id == "rec_stub0001"
    assert merged.doc_key == "t:deyne-2019-swow-en"
    assert merged.created_at == "2026-05-01T00:00:00Z"
    # fuller (file) content adopted, no longer provisional:
    assert "Small World of Words" in merged.summary
    assert merged.content_hash == "abc123"
    assert merged.source.endswith("deyne2019.pdf")
    assert merged.provisional is False
    assert merged.title.startswith("The Small World of Words")
    assert merged.kind == "index_card"


def test_merge_two_stubs_keeps_survivor_content():
    incoming_stub = _incoming_full(provisional=True, content_hash=None, source=None)
    merged = merge_card_record(_stub(), incoming_stub)
    assert "SWOW-EN" in merged.summary  # survivor's body kept
    assert merged.provisional is True


# -- repoint_staged_doc_key ------------------------------------------------

def test_repoint_staged_claims_only():
    repo = _repo_with([])
    repo.stage_record(Record(
        id="rec_a", source_uri="qmd://x/y.md",
        anchor_offset_start=0, anchor_offset_end=1,
        summary="a claim about SWOW", created_at=now_iso(),
        kind="claim", doc_key="t:deyne-2019-swow-en", title="De Deyne SWOW",
    ))
    repo.stage_record(Record(
        id="rec_b", source_uri="qmd://x/z.md",
        anchor_offset_start=0, anchor_offset_end=1,
        summary="an unrelated claim", created_at=now_iso(), kind="claim",
    ))
    # a staged CARD with the old key must NOT be repointed (claims only)
    repo.stage_record(Record(
        id="rec_c", source_uri="doc://t:deyne-2019-swow-en",
        anchor_offset_start=0, anchor_offset_end=0,
        summary="stub", created_at=now_iso(),
        kind="index_card", doc_key="t:deyne-2019-swow-en", provisional=True,
    ))

    n = repo.repoint_staged_doc_key(
        "t:deyne-2019-swow-en", "doi:10.3758/s13428-018-1115-7")
    assert n == 1  # only rec_a
    assert repo.get_staged("rec_a").doc_key == "doi:10.3758/s13428-018-1115-7"
    assert repo.get_staged("rec_b").doc_key is None
    assert repo.get_staged("rec_c").doc_key == "t:deyne-2019-swow-en"


def test_repoint_noop_same_key():
    repo = _repo_with([])
    assert repo.repoint_staged_doc_key("k", "k") == 0


# -- get_card_by_content_hash --------------------------------------------

def test_get_card_by_content_hash(tmp_path):
    conn = connect(tmp_path / "graph.db")
    apply_migrations(conn)
    repo = GraphRepo(conn)
    repo.create_record(Record(
        id="rec_card01", source_uri="file:///a.pdf",
        anchor_offset_start=0, anchor_offset_end=0,
        summary="card", created_at=now_iso(),
        kind="index_card", doc_key="doi:10.1/x",
        source="file:///a.pdf", content_hash="hh1", title="X",
    ))
    got = repo.get_card_by_content_hash("hh1")
    assert got is not None and got.id == "rec_card01"
    assert repo.get_card_by_content_hash("nope") is None
    assert repo.get_card_by_content_hash("") is None


# -- find_card_candidates (the embedding pre-filter) ---------------------

class _FakeVec:
    """Drop-in for ``RecordsVecIndex.search`` — returns canned, score-ordered
    hits so the filter logic (cards-only, threshold, exclude-self, k cap) is
    tested deterministically without fastembed."""

    def __init__(self, hits: list[VecHit]) -> None:
        self._hits = hits

    def search(self, query_text: str, k: int) -> list[VecHit]:
        return self._hits[:k]

    def add_record(self, record_id: str, summary: str) -> None:
        pass


def _repo_with(records: list[Record]) -> GraphRepo:
    conn = connect(":memory:")
    apply_migrations(conn)
    repo = GraphRepo(conn)
    for r in records:
        repo.create_record(r)
    return repo


def _card(rid: str, key: str, prov: bool = True) -> Record:
    return Record(
        id=rid, source_uri=f"doc://{key}",
        anchor_offset_start=0, anchor_offset_end=0,
        summary=f"body of {key}", created_at=now_iso(),
        kind="index_card", doc_key=key, provisional=prov,
    )


def _claim(rid: str, key: str | None = None) -> Record:
    return Record(
        id=rid, source_uri="qmd://x/y.md",
        anchor_offset_start=0, anchor_offset_end=1,
        summary=f"claim {rid}", created_at=now_iso(),
        kind="claim", doc_key=key,
    )


def test_find_candidates_filters_to_cards_above_threshold():
    repo = _repo_with([
        _card("rec_deyne", "t:deyne"),
        _card("rec_wulff", "t:wulff"),
        _claim("rec_claim", key="t:deyne"),  # a claim referencing a doc — NOT a candidate
    ])
    vec = _FakeVec([
        VecHit("rec_claim", 0.95, "claim"),   # highest score but a claim -> dropped
        VecHit("rec_deyne", 0.82, "body"),    # card, above threshold -> kept
        VecHit("rec_wulff", 0.40, "body"),    # card, below threshold -> dropped
    ])
    out = find_card_candidates(
        "incoming body", vec_index=vec, repo=repo, threshold=0.6,
    )
    assert [c.record.id for c in out] == ["rec_deyne"]
    assert isinstance(out[0], CardCandidate)
    assert out[0].score == 0.82


def test_find_candidates_excludes_self_doc_key():
    repo = _repo_with([_card("rec_deyne", "t:deyne")])
    vec = _FakeVec([VecHit("rec_deyne", 0.99, "body")])
    # a re-ingest whose own key is already carded must not match itself
    out = find_card_candidates(
        "x", vec_index=vec, repo=repo, threshold=0.6, exclude_doc_key="t:deyne",
    )
    assert out == []


def test_find_candidates_orders_desc_and_caps_k():
    repo = _repo_with([
        _card("rec_a", "t:a"), _card("rec_b", "t:b"), _card("rec_c", "t:c"),
    ])
    vec = _FakeVec([
        VecHit("rec_b", 0.70, "b"),
        VecHit("rec_a", 0.90, "a"),
        VecHit("rec_c", 0.80, "c"),
    ])
    out = find_card_candidates(
        "x", vec_index=vec, repo=repo, threshold=0.6, k=2,
    )
    assert [c.record.id for c in out] == ["rec_a", "rec_c"]  # desc, capped at 2


def test_find_candidates_empty_body_returns_empty():
    repo = _repo_with([_card("rec_a", "t:a")])
    vec = _FakeVec([VecHit("rec_a", 0.99, "a")])
    assert find_card_candidates("   ", vec_index=vec, repo=repo, threshold=0.6) == []


def test_find_candidates_no_match_returns_empty():
    repo = _repo_with([_card("rec_a", "t:a")])
    vec = _FakeVec([VecHit("rec_a", 0.50, "a")])  # below threshold
    assert find_card_candidates("x", vec_index=vec, repo=repo, threshold=0.6) == []


def test_card_candidate_threshold_in_calibrated_gap():
    # the calibrated cutoff sits in the empirical (0.56, 0.80) separation gap
    assert 0.56 < CARD_CANDIDATE_THRESHOLD < 0.80


# -- plan_card_merges -----------------------------------------------------

def _incoming(doc_key: str, *, content_hash=None, body="some body", title="T",
              provisional=False, rid="rec_inc") -> Record:
    return Record(
        id=rid, source_uri=f"doc://{doc_key}",
        anchor_offset_start=0, anchor_offset_end=0,
        summary=body, created_at=now_iso(), kind="index_card",
        doc_key=doc_key, source=None, content_hash=content_hash,
        title=title, provisional=provisional,
    )


def test_plan_skips_same_key_card():
    # a card whose doc_key already exists is finalize's job, not reconcile's
    repo = _repo_with([_card("rec_x", "t:known")])
    vec = _FakeVec([])
    plan = plan_card_merges(
        [_incoming("t:known")], repo=repo, vec_index=vec, threshold=0.6,
    )
    assert plan == {"auto_merges": [], "judge_pairs": []}


def test_plan_content_hash_auto_merge():
    repo = _repo_with([
        Record(
            id="rec_surv", source_uri="file:///a.pdf",
            anchor_offset_start=0, anchor_offset_end=0, summary="surv",
            created_at=now_iso(), kind="index_card", doc_key="t:deyne",
            source="file:///a.pdf", content_hash="HH", title="X",
        ),
    ])
    vec = _FakeVec([])
    plan = plan_card_merges(
        [_incoming("doi:10.1/x", content_hash="HH")],
        repo=repo, vec_index=vec, threshold=0.6,
    )
    assert plan["judge_pairs"] == []
    assert len(plan["auto_merges"]) == 1
    am = plan["auto_merges"][0]
    assert am["survivor_id"] == "rec_surv"
    assert am["incoming_doc_key"] == "doi:10.1/x"
    assert am["incoming_id"] == "rec_inc"
    assert am["via"] == "content_hash"


def test_plan_embedding_emits_judge_pair():
    repo = _repo_with([_card("rec_stub", "t:deyne")])
    vec = _FakeVec([VecHit("rec_stub", 0.81, "stub body")])
    plan = plan_card_merges(
        [_incoming("doi:10.1/x", body="incoming body")],
        repo=repo, vec_index=vec, threshold=0.6,
    )
    assert plan["auto_merges"] == []
    assert len(plan["judge_pairs"]) == 1
    jp = plan["judge_pairs"][0]
    assert jp["pair_id"] == 0
    assert jp["incoming_doc_key"] == "doi:10.1/x"
    assert jp["incoming_id"] == "rec_inc"
    assert jp["survivor_id"] == "rec_stub"
    assert jp["incoming_body"] == "incoming body"
    assert jp["score"] == 0.81


def test_plan_no_candidate_emits_nothing():
    repo = _repo_with([_card("rec_stub", "t:deyne")])
    vec = _FakeVec([VecHit("rec_stub", 0.30, "stub body")])  # below threshold
    plan = plan_card_merges(
        [_incoming("doi:10.1/x")], repo=repo, vec_index=vec, threshold=0.6,
    )
    assert plan == {"auto_merges": [], "judge_pairs": []}


# -- resolve_merges -------------------------------------------------------

def test_resolve_takes_auto_and_first_yes():
    plan = {
        "auto_merges": [
            {"incoming_id": "rec_ia", "incoming_doc_key": "doi:a",
             "survivor_id": "rec_sa", "survivor_doc_key": "t:a", "via": "content_hash"},
        ],
        "judge_pairs": [
            {"pair_id": 0, "incoming_id": "rec_ib", "incoming_doc_key": "doi:b",
             "survivor_id": "rec_sb1", "survivor_doc_key": "t:b1"},
            {"pair_id": 1, "incoming_id": "rec_ib", "incoming_doc_key": "doi:b",
             "survivor_id": "rec_sb2", "survivor_doc_key": "t:b2"},
        ],
    }
    # pair 0 = NO, pair 1 = YES -> incoming doi:b merges into the pair-1 survivor
    merges = resolve_merges(plan, {0: False, 1: True})
    assert len(merges) == 2
    assert merges[0]["survivor_id"] == "rec_sa" and merges[0]["via"] == "content_hash"
    b = next(m for m in merges if m["incoming_doc_key"] == "doi:b")
    assert b["survivor_id"] == "rec_sb2" and b["via"] == "judge"


def test_resolve_first_yes_wins_over_later_yes():
    plan = {"auto_merges": [], "judge_pairs": [
        {"pair_id": 0, "incoming_id": "rec_ib", "incoming_doc_key": "doi:b",
         "survivor_id": "rec_hi", "survivor_doc_key": "t:hi"},
        {"pair_id": 1, "incoming_id": "rec_ib", "incoming_doc_key": "doi:b",
         "survivor_id": "rec_lo", "survivor_doc_key": "t:lo"},
    ]}
    merges = resolve_merges(plan, {0: True, 1: True})
    assert len(merges) == 1  # one incoming -> one survivor, the first YES
    assert merges[0]["survivor_id"] == "rec_hi"


def test_resolve_missing_verdict_is_no():
    plan = {"auto_merges": [], "judge_pairs": [
        {"pair_id": 0, "incoming_id": "rec_ib", "incoming_doc_key": "doi:b",
         "survivor_id": "rec_s", "survivor_doc_key": "t:s"},
    ]}
    assert resolve_merges(plan, {}) == []


def test_resolve_survivor_conflict_skipped():
    # two incomings both confirmed into the SAME survivor -> only the first
    plan = {"auto_merges": [], "judge_pairs": [
        {"pair_id": 0, "incoming_id": "rec_ia", "incoming_doc_key": "doi:a",
         "survivor_id": "rec_s", "survivor_doc_key": "t:s"},
        {"pair_id": 1, "incoming_id": "rec_ib", "incoming_doc_key": "doi:b",
         "survivor_id": "rec_s", "survivor_doc_key": "t:s"},
    ]}
    merges = resolve_merges(plan, {0: True, 1: True})
    assert len(merges) == 1
    assert merges[0]["incoming_doc_key"] == "doi:a"


# -- apply_card_merges (the De Deyne upgrade, end to end on rows) ----------

def test_apply_merges_upgrades_survivor_and_repoints():
    # survivor stub in records
    survivor = Record(
        id="rec_stub", source_uri="doc://t:deyne",
        anchor_offset_start=0, anchor_offset_end=0,
        summary="SWOW-EN stub [unverified]", created_at="2026-05-01T00:00:00Z",
        kind="index_card", doc_key="t:deyne", source=None,
        content_hash=None, title="De Deyne SWOW-EN", provisional=True,
    )
    repo = _repo_with([survivor])
    # incoming full card (different key) staged this cycle
    repo.stage_record(Record(
        id="rec_full", source_uri="file:///deyne.pdf",
        anchor_offset_start=0, anchor_offset_end=0,
        summary="## Summary\nThe Small World of Words English norms.\n\n## Key points\n- 12k cues",
        created_at="2026-06-01T00:00:00Z", kind="index_card",
        doc_key="doi:10.3758/x", source="file:///deyne.pdf",
        content_hash="HH",
        title="The Small World of Words English word association norms",
        provisional=False,
    ))
    # a staged claim built on the incoming (carries the absorbed key)
    repo.stage_record(Record(
        id="rec_claim", source_uri="qmd://x/y.md",
        anchor_offset_start=0, anchor_offset_end=1,
        summary="a claim about SWOW", created_at="2026-06-01T00:00:00Z",
        kind="claim", doc_key="doi:10.3758/x", title="De Deyne SWOW",
    ))

    merges = [{
        "incoming_id": "rec_full", "incoming_doc_key": "doi:10.3758/x",
        "survivor_id": "rec_stub", "survivor_doc_key": "t:deyne", "via": "judge",
    }]
    embedded: list[tuple[str, str]] = []

    class _RecVec(_FakeVec):
        def add_record(self, record_id, summary):
            embedded.append((record_id, summary))

    metrics = apply_card_merges(merges, repo=repo, vec_index=_RecVec([]))

    assert metrics["merged"] == 1 and metrics["judged"] == 1
    assert metrics["claims_repointed"] == 1
    # the staged incoming card is gone (no duplicate node will be promoted)
    assert repo.get_staged("rec_full") is None
    # survivor row upgraded in place: survivor identity, fuller file content
    upgraded = repo.get_record("rec_stub")
    assert upgraded.doc_key == "t:deyne"           # survivor key kept
    assert upgraded.content_hash == "HH"           # file content adopted
    assert upgraded.provisional is False
    assert "Small World of Words" in upgraded.summary
    # body changed -> survivor re-embedded
    assert embedded and embedded[0][0] == "rec_stub"
    # the staged claim was re-pointed onto the survivor key
    assert repo.get_staged("rec_claim").doc_key == "t:deyne"


def test_apply_skips_when_survivor_missing():
    repo = _repo_with([])  # no survivor row
    repo.stage_record(_incoming("doi:x", content_hash="HH", rid="rec_f"))
    merges = [{
        "incoming_id": "rec_f", "incoming_doc_key": "doi:x",
        "survivor_id": "rec_gone", "survivor_doc_key": "t:gone", "via": "judge",
    }]
    metrics = apply_card_merges(merges, repo=repo, vec_index=None)
    assert metrics["merged"] == 0 and metrics["skipped"] == 1
    assert repo.get_staged("rec_f") is not None  # not deleted — merge not applied