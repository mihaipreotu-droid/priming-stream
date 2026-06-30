"""v0.7-x GraphRepo: records CRUD + sleep_cycles lifecycle."""
from __future__ import annotations

from priming_stream.core.db import connect
from priming_stream.core.graph_repo import GraphRepo
from priming_stream.core.models import Record, new_record_id, now_iso
from priming_stream.core.schema import apply_migrations


def _repo(tmp_path) -> GraphRepo:
    conn = connect(tmp_path / "graph.db")
    apply_migrations(conn)
    return GraphRepo(conn)


def _record(**kw) -> Record:
    defaults = dict(
        id=new_record_id(),
        source_uri="qmd://priming-stream-imports/x/y/z.md",
        anchor_offset_start=0,
        anchor_offset_end=128,
        summary="first record",
        created_at=now_iso(),
    )
    defaults.update(kw)
    return Record(**defaults)


# -- new_record_id --------------------------------------------------------

def test_new_record_id_shape():
    rid = new_record_id()
    assert rid.startswith("rec_")
    assert len(rid) == 12  # 'rec_' + 8 hex
    int(rid[4:], 16)  # raises if non-hex


def test_new_record_id_unique():
    ids = {new_record_id() for _ in range(1000)}
    assert len(ids) == 1000


# -- create / get round-trip ---------------------------------------------

def test_create_and_get_record(tmp_path):
    repo = _repo(tmp_path)
    r = _record(summary="hello world", anchor_offset_start=10,
                anchor_offset_end=42)
    repo.create_record(r)
    fetched = repo.get_record(r.id)
    assert fetched is not None
    assert fetched.id == r.id
    assert fetched.source_uri == r.source_uri
    assert fetched.anchor_offset_start == 10
    assert fetched.anchor_offset_end == 42
    assert fetched.summary == "hello world"
    assert fetched.created_at == r.created_at


def test_get_missing_record_returns_none(tmp_path):
    repo = _repo(tmp_path)
    assert repo.get_record("rec_deadbeef") is None


def test_create_record_with_null_anchors(tmp_path):
    repo = _repo(tmp_path)
    r = _record(anchor_offset_start=None, anchor_offset_end=None)
    repo.create_record(r)
    fetched = repo.get_record(r.id)
    assert fetched is not None
    assert fetched.anchor_offset_start is None
    assert fetched.anchor_offset_end is None


# -- piece3: claim defaults + index_card round-trip ----------------------

def test_claim_record_has_default_doc_fields(tmp_path):
    """A claim built without piece3 kwargs round-trips as kind='claim' with
    all document fields None — nothing expects them on a claim."""
    repo = _repo(tmp_path)
    r = _record(summary="a claim")
    repo.create_record(r)
    fetched = repo.get_record(r.id)
    assert fetched is not None
    assert fetched.kind == "claim"
    assert fetched.doc_key is None
    assert fetched.source is None
    assert fetched.content_hash is None


def test_index_card_round_trip(tmp_path):
    repo = _repo(tmp_path)
    card = _record(
        id="rec_card0001",
        source_uri="file:///C:/papers/x.md",
        summary="summary\n\nkey points\n\nrelevance",
        kind="index_card",
        doc_key="doi:10.1/abc",
        source="file:///C:/papers/x.md",
        content_hash="hash-v1",
    )
    repo.create_record(card)
    fetched = repo.get_record("rec_card0001")
    assert fetched is not None
    assert fetched.kind == "index_card"
    assert fetched.doc_key == "doi:10.1/abc"
    assert fetched.source == "file:///C:/papers/x.md"
    assert fetched.content_hash == "hash-v1"


def test_get_record_by_doc_key(tmp_path):
    repo = _repo(tmp_path)
    card = _record(id="rec_card0001", kind="index_card",
                   doc_key="path:/a/b.pdf", content_hash="h1")
    repo.create_record(card)
    got = repo.get_record_by_doc_key("path:/a/b.pdf")
    assert got is not None
    assert got.id == "rec_card0001"
    assert repo.get_record_by_doc_key("path:/nope.pdf") is None


def test_index_card_title_provisional_round_trip(tmp_path):
    repo = _repo(tmp_path)
    card = _record(id="rec_stub0001", kind="index_card",
                   doc_key="doi:10.1/x", content_hash=None,
                   title="Spreading Activation Theory", provisional=True)
    repo.create_record(card)
    got = repo.get_record("rec_stub0001")
    assert got.title == "Spreading Activation Theory"
    assert got.provisional is True
    # a full card defaults provisional False
    repo.create_record(_record(id="rec_full0001", kind="index_card",
                               doc_key="doi:10.1/y", title="X"))
    assert repo.get_record("rec_full0001").provisional is False


def test_get_record_by_doc_key_ignores_claim_sharing_key(tmp_path):
    """piece3-B: a claim may reference a doc via doc_key; the lookup must
    return the index_card, never the referencing claim."""
    repo = _repo(tmp_path)
    # a claim that references the doc (non-unique doc_key)
    repo.create_record(_record(id="rec_claimref", kind="claim",
                               doc_key="doi:10.1/x", summary="a claim about it"))
    # no card yet -> None (the claim must not be mistaken for the card)
    assert repo.get_record_by_doc_key("doi:10.1/x") is None
    # now add the card
    repo.create_record(_record(id="rec_card0009", kind="index_card",
                               doc_key="doi:10.1/x", title="X"))
    got = repo.get_record_by_doc_key("doi:10.1/x")
    assert got is not None and got.id == "rec_card0009"


def test_get_record_by_doc_key_ignores_claims(tmp_path):
    """Claims have NULL doc_key — they are invisible to doc_key lookup."""
    repo = _repo(tmp_path)
    repo.create_record(_record(id="rec_claim001", summary="claim"))
    assert repo.get_record_by_doc_key("") is None


def test_delete_record(tmp_path):
    repo = _repo(tmp_path)
    r = _record(id="rec_del00001")
    repo.create_record(r)
    assert repo.get_record("rec_del00001") is not None
    repo.delete_record("rec_del00001")
    assert repo.get_record("rec_del00001") is None
    # idempotent — deleting an absent id is a no-op
    repo.delete_record("rec_del00001")


# -- list_records ---------------------------------------------------------

def test_list_records_most_recent_first(tmp_path):
    repo = _repo(tmp_path)
    a = _record(id="rec_00000001", created_at="2026-05-01T00:00:00Z",
                summary="oldest")
    b = _record(id="rec_00000002", created_at="2026-05-02T00:00:00Z",
                summary="middle")
    c = _record(id="rec_00000003", created_at="2026-05-03T00:00:00Z",
                summary="newest")
    # Insert out of timestamp order to prove ordering is by created_at.
    repo.create_record(b)
    repo.create_record(a)
    repo.create_record(c)

    listed = repo.list_records()
    assert [r.id for r in listed] == [c.id, b.id, a.id]


def test_list_records_limit(tmp_path):
    repo = _repo(tmp_path)
    for i in range(5):
        repo.create_record(
            _record(id=f"rec_0000000{i}",
                    created_at=f"2026-05-0{i+1}T00:00:00Z",
                    summary=f"r{i}"),
        )
    listed = repo.list_records(limit=2)
    assert len(listed) == 2
    assert listed[0].summary == "r4"
    assert listed[1].summary == "r3"


def test_list_records_empty(tmp_path):
    repo = _repo(tmp_path)
    assert repo.list_records() == []
    assert repo.list_records(limit=10) == []


# -- records_by_source_uri ------------------------------------------------

def test_records_by_source_uri_prefix(tmp_path):
    repo = _repo(tmp_path)
    qmd_a = _record(
        id="rec_qmd00001",
        source_uri="qmd://priming-stream-imports/a/b.md",
        created_at="2026-05-01T00:00:00Z",
    )
    qmd_b = _record(
        id="rec_qmd00002",
        source_uri="qmd://priming-stream-imports/a/c.md",
        created_at="2026-05-02T00:00:00Z",
    )
    file_a = _record(
        id="rec_file0001",
        source_uri="file:///C:/x/y.md",
        created_at="2026-05-03T00:00:00Z",
    )
    repo.create_record(qmd_a)
    repo.create_record(qmd_b)
    repo.create_record(file_a)

    hits = repo.records_by_source_uri("qmd://priming-stream-imports/")
    assert {r.id for r in hits} == {qmd_a.id, qmd_b.id}
    # ordering: most recent first
    assert hits[0].id == qmd_b.id


def test_records_by_source_uri_no_match(tmp_path):
    repo = _repo(tmp_path)
    repo.create_record(_record(source_uri="file:///x.md"))
    assert repo.records_by_source_uri("qmd://nope/") == []


# -- sleep cycles ---------------------------------------------------------

def test_sleep_cycle_lifecycle(tmp_path):
    repo = _repo(tmp_path)
    cycle_id = repo.start_sleep_cycle(started_at="2026-05-25T10:00:00Z")
    assert isinstance(cycle_id, int)

    repo.finish_sleep_cycle(
        cycle_id,
        completed_at="2026-05-25T10:05:00Z",
        chunks_materialized=12,
        records_created=7,
        records_skipped=5,
        metrics_json='{"phase_a_ms": 42}',
        notes="ok",
    )

    cycles = repo.list_sleep_cycles()
    assert len(cycles) == 1
    row = cycles[0]
    assert row["id"] == cycle_id
    assert row["started_at"] == "2026-05-25T10:00:00Z"
    assert row["completed_at"] == "2026-05-25T10:05:00Z"
    assert row["chunks_materialized"] == 12
    assert row["records_created"] == 7
    assert row["records_skipped"] == 5
    assert row["metrics_json"] == '{"phase_a_ms": 42}'
    assert row["notes"] == "ok"


def test_list_sleep_cycles_recent_first(tmp_path):
    repo = _repo(tmp_path)
    a = repo.start_sleep_cycle(started_at="2026-05-25T10:00:00Z")
    b = repo.start_sleep_cycle(started_at="2026-05-25T11:00:00Z")
    c = repo.start_sleep_cycle(started_at="2026-05-25T12:00:00Z")
    cycles = repo.list_sleep_cycles()
    assert [row["id"] for row in cycles] == [c, b, a]


def test_list_sleep_cycles_limit(tmp_path):
    repo = _repo(tmp_path)
    for i in range(5):
        repo.start_sleep_cycle(started_at=f"2026-05-25T1{i}:00:00Z")
    cycles = repo.list_sleep_cycles(limit=2)
    assert len(cycles) == 2


def test_finish_sleep_cycle_nullable_notes(tmp_path):
    repo = _repo(tmp_path)
    cid = repo.start_sleep_cycle(started_at=now_iso())
    repo.finish_sleep_cycle(
        cid,
        completed_at=now_iso(),
        chunks_materialized=0,
        records_created=0,
        records_skipped=0,
        metrics_json="{}",
        notes=None,
    )
    cycles = repo.list_sleep_cycles()
    assert cycles[0]["notes"] is None
