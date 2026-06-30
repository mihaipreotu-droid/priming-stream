"""Unit — synthetic fixture sanity: format contract + expected_chunk alignment."""
import json
from pathlib import Path

FIX = Path(__file__).parent.parent / "fixtures"


def test_synthetic_session_format():
    lines = (FIX / "synthetic_session.jsonl").read_text(
        encoding="utf-8").splitlines()
    assert len(lines) == 6
    for raw in lines:
        rec = json.loads(raw)
        assert rec["type"] in ("user", "assistant")
        assert rec["sessionId"] == "sess-synthetic-001"
        assert rec["timestamp"].endswith("Z")
        msg = rec["message"]
        if rec["type"] == "user":
            assert isinstance(msg["content"], str)
        else:
            assert isinstance(msg["content"], list)
            assert msg["content"][0]["type"] == "text"


def test_idle_gap_present():
    recs = [json.loads(l) for l in
            (FIX / "synthetic_session.jsonl").read_text(
                encoding="utf-8").splitlines()]
    # turn 3 -> turn 4 gap exceeds 30 min (burst split point)
    assert recs[3]["timestamp"] == "2026-05-20T10:06:00Z"
    assert recs[4]["timestamp"] == "2026-05-20T10:50:00Z"


def test_expected_chunk_matches_first_burst():
    expected = json.loads(
        (FIX / "expected_chunk.json").read_text(encoding="utf-8"))
    recs = [json.loads(l) for l in
            (FIX / "synthetic_session.jsonl").read_text(
                encoding="utf-8").splitlines()]

    assert expected["chunk_id"] == "claude_code_sess-synthetic-001_000"
    assert expected["source_client"] == "claude_code"
    assert expected["session_id"] == "sess-synthetic-001"
    assert len(expected["turns"]) == 4

    def text_of(rec):
        c = rec["message"]["content"]
        return c if isinstance(c, str) else c[0]["text"]

    for i, turn in enumerate(expected["turns"]):
        assert turn["index"] == i
        assert turn["role"] == recs[i]["type"]
        assert turn["text"] == text_of(recs[i])
        assert turn["timestamp"] == recs[i]["timestamp"]

    assert expected["started_at"] == recs[0]["timestamp"]
    assert expected["ended_at"] == recs[3]["timestamp"]
