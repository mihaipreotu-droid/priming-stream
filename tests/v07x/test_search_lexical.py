"""Tests for on-demand lexical search (graph_ops.graph_search_lexical)."""
from __future__ import annotations

import pytest

from priming_stream.core.db import connect
from priming_stream.core.graph_repo import GraphRepo
from priming_stream.core.models import Record, now_iso
from priming_stream.core.schema import apply_migrations
from priming_stream.graph_ops.records_search import _fts5_match, graph_search_lexical


# -- _fts5_match -----------------------------------------------------------


def test_fts5_match_and_or_phrase():
    assert _fts5_match("Dirichlet calibration", "and") == '"Dirichlet" AND "calibration"'
    assert _fts5_match("Dirichlet calibration", "or") == '"Dirichlet" OR "calibration"'
    assert _fts5_match("spreading activation", "phrase") == '"spreading activation"'


def test_fts5_match_drops_single_char_and_empty():
    # single-char tokens dropped (bm25 noise); punctuation tokenized out
    assert _fts5_match("a Dirichlet b", "and") == '"Dirichlet"'
    assert _fts5_match("", "and") == ""
    assert _fts5_match("   ", "or") == ""
    assert _fts5_match("!! ?", "and") == ""


# -- graph_search_lexical --------------------------------------------------


@pytest.fixture()
def repo(tmp_path):
    conn = connect(tmp_path / "graph.db")
    apply_migrations(conn)
    r = GraphRepo(conn)
    recs = [
        ("rec_a", "Dirichlet prior on edge weights for Meshgraph calibration"),
        ("rec_b", "Collins and Loftus 1975 spreading activation theory"),
        ("rec_c", "PanelX pilot 76 respondenti 305 noduri"),
    ]
    for rid, summary in recs:
        r.create_record(Record(
            id=rid, source_uri=f"qmd://x/{rid}.md",
            anchor_offset_start=0, anchor_offset_end=1,
            summary=summary, created_at=now_iso(),
            source_date="2026-05-01T10:00:00Z",
        ))
    yield r
    conn.close()


def _ids(results):
    return {r["record_id"] for r in results}


def test_and_requires_all_terms(repo):
    # both terms in rec_a only
    assert _ids(graph_search_lexical("Dirichlet calibration", 10, "and", repo)) == {"rec_a"}
    # no record has BOTH Dirichlet and Collins
    assert graph_search_lexical("Dirichlet Collins", 10, "and", repo) == []


def test_or_recall(repo):
    assert _ids(graph_search_lexical("Dirichlet Collins", 10, "or", repo)) == {"rec_a", "rec_b"}


def test_phrase_is_ordered(repo):
    assert _ids(graph_search_lexical("spreading activation", 10, "phrase", repo)) == {"rec_b"}
    # wrong order → no phrase match
    assert graph_search_lexical("activation spreading", 10, "phrase", repo) == []


def test_unknown_mode_falls_back_to_and(repo):
    assert _ids(graph_search_lexical("Dirichlet calibration", 10, "bogus", repo)) == {"rec_a"}


def test_empty_and_no_match(repo):
    assert graph_search_lexical("", 10, "and", repo) == []
    assert graph_search_lexical("zzzznonexistentterm", 10, "and", repo) == []


def test_return_shape(repo):
    [hit] = graph_search_lexical("Dirichlet", 10, "and", repo)
    assert hit["record_id"] == "rec_a"
    assert set(hit) == {"record_id", "summary", "score", "source_uri", "source_date", "kind"}
    assert hit["source_date"] == "2026-05-01T10:00:00Z"
    assert hit["kind"] == "claim"
    assert isinstance(hit["score"], float)


def test_k_limits(repo):
    # OR over a common-ish term set; k=1 returns at most 1
    out = graph_search_lexical("Dirichlet Collins PanelX", 1, "or", repo)
    assert len(out) == 1


def test_diacritics_tokenize(repo):
    # Romanian diacritics tokenize via unicode61
    assert _ids(graph_search_lexical("respondenti", 10, "and", repo)) == {"rec_c"}
