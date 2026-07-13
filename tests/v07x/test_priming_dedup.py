"""Item 3.3 — cross-turn priming dedup (sliding turn window, N default 10).

Covers the mechanism end to end at the unit level:

* ``select_semantic`` drops ``exclude_recent_ids`` BEFORE truncation so the
  freed budget backfills from the recency-ranked tail (the queue advances);
* ``lexical_bucket`` drops recent ids too, but WITHOUT inflating its over-fetch
  by the (large, mostly off-topic) recent set — the turn-69 regression: an
  index_card that only exists in the enlarged window must not be pulled in and
  kind-biased ahead of genuine hits, perturbing NON-recent records;
* ``build_priming`` threads the set into both buckets;
* the hook's ``_recent_primed_ids`` reconstructs the per-session window from the
  tail of ``echoes.jsonl`` (session-scoped, last-N, kill-switch aware).

All defaults preserve prior behaviour: empty ``exclude_recent_ids`` is a no-op.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

from priming_stream.bridge.lexical import lexical_bucket
from priming_stream.bridge.recency import select_semantic
from priming_stream.bridge.types import ScoredRecord
from priming_stream.bridge.working_set import build_priming
from priming_stream.core.db import connect
from priming_stream.core.graph_repo import GraphRepo
from priming_stream.core.models import Record
from priming_stream.core.schema import apply_migrations


_NOW = datetime(2026, 6, 4, 12, 0, 0, tzinfo=timezone.utc)


def _cfg(**overrides):
    base = dict(
        decay=0.8, min_score=0.3, frontier_cap=10, k_per_query=30, max_hops=4,
        max_records=20, recency_strength=0.25, recency_age_span_days=180,
        recency_p_max=0.5, bucket_total=25, bucket_lexical=5,
        recency_filter_cutoff="",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _sr(rid: str, score: float) -> ScoredRecord:
    return ScoredRecord(
        record=Record(
            id=rid, source_uri=f"qmd://c/{rid}.md",
            anchor_offset_start=0, anchor_offset_end=0,
            summary=f"summary {rid}", created_at="2026-06-01T00:00:00Z",
        ),
        score=score,
    )


def _conn(tmp_path):
    conn = connect(tmp_path / "graph.db")
    apply_migrations(conn)
    return conn


def _add(conn, summary, *, kind="claim", rid=None):
    from priming_stream.core.models import new_record_id, now_iso
    rid = rid or new_record_id()
    conn.execute(
        "INSERT INTO records (id, source_uri, anchor_offset_start, "
        "anchor_offset_end, summary, created_at, source_date, kind, doc_key, "
        "source, content_hash, title, provisional) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (rid, "qmd://t/r.md" if kind == "claim" else "file:///d.pdf",
         0, len(summary), summary, now_iso(), None, kind,
         "t:doc" if kind == "index_card" else None, None, None,
         "Doc" if kind == "index_card" else None, 0),
    )
    conn.commit()
    return rid


# ------------------------------------------------------- select_semantic


def test_select_semantic_empty_exclude_is_noop():
    activated = [_sr(f"rec_{i:02d}", 0.9 - i * 0.01) for i in range(25)]
    base = select_semantic(activated, _cfg(), now=_NOW)
    with_empty = select_semantic(
        activated, _cfg(), now=_NOW, exclude_recent_ids=frozenset()
    )
    assert [s.record.id for s in base] == [s.record.id for s in with_empty]


def test_select_semantic_filters_before_truncation_and_backfills():
    # 25 candidates, budget = bucket_total(25) - bucket_lexical(5) = 20.
    activated = [_sr(f"rec_{i:02d}", 0.9 - i * 0.01) for i in range(25)]
    baseline = select_semantic(activated, _cfg(), now=_NOW)
    assert [s.record.id for s in baseline] == [f"rec_{i:02d}" for i in range(20)]

    # Suppress the top 3; freed slots must backfill from the tail (20,21,22),
    # NOT shrink to 17. The queue advances.
    recent = frozenset({"rec_00", "rec_01", "rec_02"})
    out = select_semantic(activated, _cfg(), now=_NOW, exclude_recent_ids=recent)
    ids = [s.record.id for s in out]
    assert len(ids) == 20                       # total preserved (deep pool)
    assert recent.isdisjoint(ids)               # suppressed ids gone
    assert {"rec_20", "rec_21", "rec_22"} <= set(ids)  # tail backfilled in
    assert "rec_23" not in ids and "rec_24" not in ids  # only 3 slots freed


def test_select_semantic_thin_pool_shrinks_never_pads():
    # Pool smaller than budget: suppression shrinks the total (correct — no
    # filler to backfill), never pads.
    activated = [_sr(f"rec_{i:02d}", 0.9 - i * 0.01) for i in range(5)]
    out = select_semantic(
        activated, _cfg(), now=_NOW,
        exclude_recent_ids=frozenset({"rec_00", "rec_01"}),
    )
    assert [s.record.id for s in out] == ["rec_02", "rec_03", "rec_04"]


# ------------------------------------------------------- lexical_bucket


def test_lexical_exclude_recent_drops_and_backfills(tmp_path):
    conn = _conn(tmp_path)
    try:
        ids = [_add(conn, f"bridge record variant {i}") for i in range(6)]
        base = lexical_bucket(conn, "bridge", limit=3, exclude_ids=set(),
                              kind_bias=False)
        base_ids = [s.record.id for s in base]
        # Suppress the top base hit → it drops, backfilled from the tail to 3.
        recent = frozenset({base_ids[0]})
        out = lexical_bucket(conn, "bridge", limit=3, exclude_ids=set(),
                             kind_bias=False, exclude_recent_ids=recent)
        out_ids = [s.record.id for s in out]
        assert base_ids[0] not in out_ids
        assert len(out_ids) == 3                # backfilled, not shrunk
    finally:
        conn.close()


def test_lexical_unrelated_recent_ids_do_not_perturb_output(tmp_path):
    """Turn-69 regression: recent ids that are NOT among the BM25 hits must not
    change WHICH non-recent records surface. Sizing the over-fetch on the recent
    set would enlarge the fetch window and let kind_bias promote an index_card
    that only exists deeper in that window — perturbing non-recent records."""
    conn = _conn(tmp_path)
    try:
        # 6 strong claim hits + 1 WEAK card hit (bridge once amid filler) → the
        # card sits at BM25 rank 7, outside the base fetch window (2*limit=6).
        for i in range(6):
            _add(conn, f"bridge bridge strong claim {i}")
        card = _add(
            conn,
            "neural overview mentioning bridge once amid lots of padding words",
            kind="index_card",
        )
        # A big recent set of ids that DON'T match 'bridge' at all.
        unrelated_recent = frozenset(
            _add(conn, f"totally unrelated pottery glaze note {j}")
            for j in range(12)
        )
        base = lexical_bucket(conn, "bridge", limit=3, exclude_ids=set(),
                              kind_bias=True)
        withr = lexical_bucket(conn, "bridge", limit=3, exclude_ids=set(),
                               kind_bias=True,
                               exclude_recent_ids=unrelated_recent)
        base_ids = [s.record.id for s in base]
        withr_ids = [s.record.id for s in withr]
        assert base_ids == withr_ids            # identical — no perturbation
        assert card not in withr_ids            # weak card did NOT sneak in
    finally:
        conn.close()


# ------------------------------------------------------- build_priming


def test_build_priming_excludes_recent_from_both_buckets(tmp_path):
    conn = _conn(tmp_path)
    repo = GraphRepo(conn)
    # semantic candidate + a lexical-only candidate, both matching 'bridge'.
    repo.create_record(Record(
        id="rec_sem00001", source_uri="qmd://c/s.md",
        anchor_offset_start=0, anchor_offset_end=0,
        summary="bridge daemon warm latency", created_at="2026-06-01T00:00:00Z",
    ))
    repo.create_record(Record(
        id="rec_lex00001", source_uri="qmd://c/l.md",
        anchor_offset_start=0, anchor_offset_end=0,
        summary="bridge lexical only tail", created_at="2026-06-01T00:00:00Z",
    ))

    class _Vec:
        def embed_texts(self, texts):
            return [[0.0, 0.0] for _ in texts]

        def embeddings_for(self, rids):
            return {r: [-1.0, 0.0] for r in rids}

        def query_by_vecs(self, vecs, k):
            from priming_stream.integrations.vec_index import VecHit
            return [[VecHit(record_id="rec_sem00001", score=0.9, summary="")]
                    for _ in vecs]

    # Baseline: semantic has the sem record.
    base = build_priming("bridge", "", vec_index=_Vec(), repo=repo, conn=conn,
                         cfg=_cfg(), now=_NOW)
    assert "rec_sem00001" in {s.record.id for s in base.semantic}

    # Suppress both → semantic drops the sem record; lexical drops the lex one.
    out = build_priming("bridge", "", vec_index=_Vec(), repo=repo, conn=conn,
                        cfg=_cfg(), now=_NOW,
                        exclude_recent_ids=frozenset(
                            {"rec_sem00001", "rec_lex00001"}))
    assert "rec_sem00001" not in {s.record.id for s in out.semantic}
    assert "rec_lex00001" not in {s.record.id for s in out.lexical}
    conn.close()


# ------------------------------------------------------- hook window read


def _write_echoes(tmp_path, rows):
    epi = tmp_path / "episodic"
    epi.mkdir(parents=True, exist_ok=True)
    with (epi / "echoes.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_recent_primed_ids_session_scoped_and_windowed(tmp_path, monkeypatch):
    from priming_stream.hooks import user_prompt_submit as hook
    monkeypatch.setenv("PRIMING_STREAM_STORAGE_DIR", str(tmp_path))
    monkeypatch.delenv("PRIMING_STREAM_DEDUP_OFF", raising=False)
    # 12 turns for s1 (each a unique id) interleaved with s2 noise; N default 10
    rows = []
    for i in range(12):
        rows.append({"session_id": "s1", "semantic": [f"rec_s1_{i:02d}"],
                     "lexical": []})
        rows.append({"session_id": "s2", "semantic": [f"rec_s2_{i:02d}"],
                     "lexical": ["rec_s2_lex"]})
    _write_echoes(tmp_path, rows)

    got = set(hook._recent_primed_ids("s1"))
    # Only the last 10 s1 turns (rec_s1_02 .. rec_s1_11); s2 excluded entirely.
    assert got == {f"rec_s1_{i:02d}" for i in range(2, 12)}
    assert not any(x.startswith("rec_s2") for x in got)


def test_recent_primed_ids_unions_both_channels(tmp_path, monkeypatch):
    from priming_stream.hooks import user_prompt_submit as hook
    monkeypatch.setenv("PRIMING_STREAM_STORAGE_DIR", str(tmp_path))
    monkeypatch.delenv("PRIMING_STREAM_DEDUP_OFF", raising=False)
    _write_echoes(tmp_path, [
        {"session_id": "s1", "semantic": ["rec_a"], "lexical": ["rec_b"]},
    ])
    assert set(hook._recent_primed_ids("s1")) == {"rec_a", "rec_b"}


def test_recent_primed_ids_kill_switch_and_no_session(tmp_path, monkeypatch):
    from priming_stream.hooks import user_prompt_submit as hook
    monkeypatch.setenv("PRIMING_STREAM_STORAGE_DIR", str(tmp_path))
    _write_echoes(tmp_path, [
        {"session_id": "s1", "semantic": ["rec_a"], "lexical": []},
    ])
    # kill-switch
    monkeypatch.setenv("PRIMING_STREAM_DEDUP_OFF", "1")
    assert hook._recent_primed_ids("s1") == []
    monkeypatch.delenv("PRIMING_STREAM_DEDUP_OFF", raising=False)
    # no session id
    assert hook._recent_primed_ids(None) == []
    assert hook._recent_primed_ids("") == []


def test_recent_primed_ids_missing_log_returns_empty(tmp_path, monkeypatch):
    from priming_stream.hooks import user_prompt_submit as hook
    monkeypatch.setenv("PRIMING_STREAM_STORAGE_DIR", str(tmp_path))
    monkeypatch.delenv("PRIMING_STREAM_DEDUP_OFF", raising=False)
    # no echoes.jsonl written
    assert hook._recent_primed_ids("s1") == []
