"""Verdict readers for the unified reconcile judge (batch_*.jsonl format).

The judge Workflow pools card + claim pairs, batches them, and writes verdicts as
JSON-lines into each pipeline's verdicts dir. These tests cover the Python side
that reads them back — card (``same``) and claim (``verdict`` + ``delete_id``) —
including malformed-line tolerance, multi-batch merge, plan-scoping, and the
card/claim pair_id namespacing.
"""
from __future__ import annotations

from priming_stream.cli.reconcile import (
    _iter_verdict_jsonl,
    _read_claim_verdicts,
    _read_verdicts,
)


def _write(dir_, name: str, *lines: str) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / name).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


# -- _iter_verdict_jsonl --------------------------------------------------

def test_iter_skips_malformed_and_blank(tmp_path):
    _write(tmp_path, "batch_0.jsonl",
           '{"kind":"card","pair_id":0,"same":true}',
           '',
           'not json at all',
           '{"kind":"claim","pair_id":1,"verdict":"distinct","delete_id":null}')
    objs = list(_iter_verdict_jsonl(tmp_path))
    assert len(objs) == 2
    assert objs[0]["pair_id"] == 0 and objs[1]["pair_id"] == 1


def test_iter_missing_dir_is_empty(tmp_path):
    assert list(_iter_verdict_jsonl(tmp_path / "nope")) == []


def test_iter_merges_multiple_batches(tmp_path):
    _write(tmp_path, "batch_0.jsonl", '{"kind":"card","pair_id":0,"same":true}')
    _write(tmp_path, "batch_1.jsonl", '{"kind":"card","pair_id":1,"same":false}')
    pids = sorted(o["pair_id"] for o in _iter_verdict_jsonl(tmp_path))
    assert pids == [0, 1]


# -- _read_verdicts (card) ------------------------------------------------

def _card_pairs(*pids):
    return [{"pair_id": p} for p in pids]


def test_read_card_verdicts_basic(tmp_path):
    _write(tmp_path, "batch_0.jsonl",
           '{"kind":"card","pair_id":0,"same":true}',
           '{"kind":"card","pair_id":1,"same":false}')
    out = _read_verdicts(tmp_path, _card_pairs(0, 1))
    assert out == {0: True, 1: False}


def test_read_card_missing_pair_absent(tmp_path):
    # pair 1 has no verdict line -> absent (resolve_merges treats missing as NO)
    _write(tmp_path, "batch_0.jsonl", '{"kind":"card","pair_id":0,"same":true}')
    out = _read_verdicts(tmp_path, _card_pairs(0, 1))
    assert out == {0: True}


def test_read_card_ignores_out_of_plan_pairs(tmp_path):
    # a stale entry for a pair_id not in this plan must be ignored
    _write(tmp_path, "batch_0.jsonl",
           '{"kind":"card","pair_id":0,"same":true}',
           '{"kind":"card","pair_id":99,"same":true}')
    out = _read_verdicts(tmp_path, _card_pairs(0))
    assert out == {0: True}


def test_read_card_ignores_claim_entries(tmp_path):
    # claim entries living in a shared file must not be read as card verdicts
    _write(tmp_path, "batch_0.jsonl",
           '{"kind":"card","pair_id":0,"same":true}',
           '{"kind":"claim","pair_id":0,"verdict":"contradiction","delete_id":"rec_x"}')
    out = _read_verdicts(tmp_path, _card_pairs(0))
    assert out == {0: True}


# -- _read_claim_verdicts -------------------------------------------------

def test_read_claim_verdicts_basic(tmp_path):
    _write(tmp_path, "batch_0.jsonl",
           '{"kind":"claim","pair_id":0,"verdict":"contradiction","delete_id":"rec_old"}',
           '{"kind":"claim","pair_id":1,"verdict":"distinct","delete_id":null}')
    out = _read_claim_verdicts(tmp_path, [{"pair_id": 0}, {"pair_id": 1}])
    assert out[0] == {"verdict": "contradiction", "delete_id": "rec_old"}
    assert out[1] == {"verdict": "distinct", "delete_id": None}


def test_read_claim_none_string_normalized(tmp_path):
    _write(tmp_path, "batch_0.jsonl",
           '{"kind":"claim","pair_id":0,"verdict":"distinct","delete_id":"none"}')
    out = _read_claim_verdicts(tmp_path, [{"pair_id": 0}])
    assert out[0]["delete_id"] is None


def test_read_claim_scoped_to_plan(tmp_path):
    _write(tmp_path, "batch_0.jsonl",
           '{"kind":"claim","pair_id":0,"verdict":"near-clone","delete_id":"rec_a"}',
           '{"kind":"claim","pair_id":7,"verdict":"contradiction","delete_id":"rec_b"}')
    out = _read_claim_verdicts(tmp_path, [{"pair_id": 0}])
    assert set(out) == {0}


def test_read_claim_ignores_card_entries(tmp_path):
    _write(tmp_path, "batch_0.jsonl",
           '{"kind":"card","pair_id":0,"same":true}')
    out = _read_claim_verdicts(tmp_path, [{"pair_id": 0}])
    assert out == {}


def test_read_claim_verdict_case_normalized(tmp_path):
    _write(tmp_path, "batch_0.jsonl",
           '{"kind":"claim","pair_id":0,"verdict":"Contradiction","delete_id":"rec_x"}')
    out = _read_claim_verdicts(tmp_path, [{"pair_id": 0}])
    assert out[0]["verdict"] == "contradiction"
