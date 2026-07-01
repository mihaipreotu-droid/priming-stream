"""v0.7-x-piece3 Phase C1: doc_ingest identity + change-detection helpers."""
from __future__ import annotations

from priming_stream.core.db import connect
from priming_stream.core.graph_repo import GraphRepo
from priming_stream.core.models import Record, new_record_id, now_iso
from priming_stream.core.schema import apply_migrations
import pytest

from priming_stream.ingest.doc_ingest import (
    canonical_doc_key,
    card_md_filename,
    content_hash,
    make_card_record,
)


# -- canonical_doc_key (piece3-B identity hierarchy) ---------------------

def test_canonical_key_doi_wins_and_normalizes():
    k = canonical_doc_key(doi="https://doi.org/10.1037/0033-295X.82.6.407",
                          authors="Collins & Loftus", year=1975, title="Spreading activation")
    assert k == "doi:10.1037/0033-295x.82.6.407"


def test_canonical_key_doi_bare_and_prefixed():
    assert canonical_doc_key(doi="10.1/ABC") == "doi:10.1/abc"
    assert canonical_doc_key(doi="doi:10.1/abc") == "doi:10.1/abc"


def test_canonical_key_url_when_no_doi():
    k = canonical_doc_key(url="https://www.example.com/report/2025/")
    assert k == "url:example.com/report/2025"


def test_canonical_key_title_slug_fallback():
    k = canonical_doc_key(authors="Collins & Loftus", year=1975,
                          title="A Spreading-Activation Theory of Semantic Processing")
    assert k.startswith("t:collins-1975-")
    assert "spreading-activation" in k


def test_canonical_key_first_author_surname():
    assert canonical_doc_key(authors="Anderson, J. R.", year="1983",
                             title="ACT").startswith("t:anderson-1983-")
    assert canonical_doc_key(authors=["Tsugawa", "Jena"], year=2017,
                             title="x").startswith("t:tsugawa-2017-")


def test_canonical_key_own_doc_via_fallback_filename():
    # No academic metadata — own/partner doc keyed off its filename.
    k = canonical_doc_key(fallback="ACME Q3 2025 Brand Report.docx")
    assert k.startswith("t:acme-q3-2025-brand-report")


def test_canonical_key_strips_citation_echo_in_title():
    """polish: a title that repeats the author/year citation must not double
    it in the key ('aeschbach-2025-aeschbach-et-al-2025-...')."""
    k = canonical_doc_key(authors="Aeschbach et al.", year=2025,
                          title="Aeschbach et al. 2025 Intelligence Association")
    assert k == "t:aeschbach-2025-intelligence-association"
    # and a normal (non-echo) title is unaffected
    k2 = canonical_doc_key(authors="Roberts", year=2018, title="Word association timing")
    assert k2 == "t:roberts-2018-word-association-timing"


def test_canonical_key_deterministic():
    a = canonical_doc_key(authors="Anderson", year=1983, title="ACT theory")
    b = canonical_doc_key(authors="Anderson", year=1983, title="ACT theory")
    assert a == b


def test_canonical_key_requires_a_component():
    with pytest.raises(ValueError):
        canonical_doc_key()


# -- content_hash ---------------------------------------------------------

def test_content_hash_stable_and_16_hex():
    h = content_hash(b"hello world")
    assert h == content_hash("hello world")  # str == bytes(utf-8)
    assert len(h) == 16
    int(h, 16)  # hex


def test_content_hash_changes_on_edit():
    assert content_hash("paper body v1") != content_hash("paper body v2")


# -- card_md_filename -----------------------------------------------------

def test_card_filename_shape():
    fn = card_md_filename("doi:10.1/abc")
    assert fn.startswith("card_")
    assert fn.endswith(".md")


def test_card_filename_deterministic_by_doc_key():
    # Same doc_key -> same filename (regen overwrites in place).
    assert card_md_filename("k1") == card_md_filename("k1")
    # Different doc_key -> different filename.
    assert card_md_filename("k1") != card_md_filename("k2")


def test_card_filename_distinct_from_claim_naming():
    # Claims are rec_*.md; cards are card_*.md — no collision in records_dir.
    assert card_md_filename("anything").startswith("card_")


# -- make_card_record (producer -> finalize contract) ---------------------

def test_make_card_record_is_well_formed_index_card():
    """The card record a producer stages MUST be a well-formed index_card
    the finalize promotion accepts."""
    rec = make_card_record(
        rec_id="rec_card0001",
        source_uri="file:///C:/Vault/wiki/surse/collins-loftus-1975.md",
        doc_key="collins-loftus-1975",
        source="file:///C:/Vault/wiki/surse/collins-loftus-1975.md",
        content_hash="abc123def456",
        created_at="2026-06-01T00:00:00Z",
        body="## Summary\nSpreading-activation theory.\n\n## Relevance\nMeshgraph core.",
    )
    assert rec.kind == "index_card"
    assert rec.doc_key == "collins-loftus-1975"
    assert rec.source.endswith("collins-loftus-1975.md")
    assert rec.content_hash == "abc123def456"
    assert rec.id == "rec_card0001"
    assert rec.anchor_offset_start == 0
    assert rec.anchor_offset_end == 0
    assert "Spreading-activation" in rec.summary


def test_make_card_record_round_trips_through_staging():
    """Body with quotes/colons survives the staging round-trip verbatim, and
    empty source/content_hash normalize to None."""
    body = '## Summary\nHe said: "uptake" is the 4th M; weights are unsigned.'
    rec = make_card_record(
        rec_id="rec_card0002", source_uri="file:///a.md",
        doc_key="k", source="", content_hash="",
        created_at="2026-06-01T00:00:00Z", body=body,
    )
    assert rec.source is None and rec.content_hash is None
    conn = connect(":memory:")
    apply_migrations(conn)
    repo = GraphRepo(conn)
    repo.stage_record(rec)
    got = repo.get_staged("rec_card0002")
    assert got is not None
    assert '"uptake"' in got.summary
    assert got.kind == "index_card"


# -- prefilter by content_hash (piece3-C canonical rewire) ---------------

def _repo(tmp_path) -> GraphRepo:
    conn = connect(tmp_path / "graph.db")
    apply_migrations(conn)
    return GraphRepo(conn)


def _card(doc_key: str, content_hash_: str) -> Record:
    return Record(
        id=new_record_id(),
        source_uri="file:///C:/x.md",
        anchor_offset_start=0,
        anchor_offset_end=0,
        summary="a card",
        created_at=now_iso(),
        kind="index_card",
        doc_key=doc_key,
        source="file:///C:/x.md",
        content_hash=content_hash_,
    )


def test_card_exists_with_content_hash(tmp_path):
    repo = _repo(tmp_path)
    assert repo.card_exists_with_content_hash("h1") is False
    repo.create_record(_card("doi:10.1/abc", "h1"))
    assert repo.card_exists_with_content_hash("h1") is True   # unchanged -> skip
    assert repo.card_exists_with_content_hash("h2") is False  # changed -> ingest


def test_card_exists_with_content_hash_ignores_claims(tmp_path):
    """A claim's content_hash (None) and a non-card row never match."""
    repo = _repo(tmp_path)
    repo.create_record(Record(
        id="rec_claim001", source_uri="qmd://x/y.md",
        anchor_offset_start=0, anchor_offset_end=1,
        summary="claim", created_at=now_iso(),
    ))
    assert repo.card_exists_with_content_hash("") is False
