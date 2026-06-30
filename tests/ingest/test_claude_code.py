"""Unit tests — Claude Code adapter and burst splitting."""
import json
from pathlib import Path

from priming_stream.ingest.base import split_bursts
from priming_stream.ingest.claude_code import ClaudeCodeAdapter
from priming_stream.core.models import Turn

FIX = Path(__file__).parent.parent / "fixtures" / "synthetic_session.jsonl"
EXPECTED = Path(__file__).parent.parent / "fixtures" / "expected_chunk.json"


def _cc_line(role, ts, text, session="sess-1", uuid="x"):
    if role == "user":
        content = text
    else:
        content = [{"type": "text", "text": text}]
    return json.dumps(
        {
            "type": role,
            "timestamp": ts,
            "sessionId": session,
            "uuid": uuid,
            "message": {"role": role, "content": content},
        }
    )


def _write(path: Path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_fixture_splits_into_two_chunks():
    chunks = list(ClaudeCodeAdapter(FIX, idle_minutes=30).iter_chunks())
    assert len(chunks) == 2
    assert [len(c.turns) for c in chunks] == [4, 2]


def test_first_chunk_matches_expected_fixture():
    chunks = list(ClaudeCodeAdapter(FIX, idle_minutes=30).iter_chunks())
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))
    c = chunks[0]
    assert c.chunk_id == expected["chunk_id"]
    assert c.source_client == expected["source_client"]
    assert c.session_id == expected["session_id"]
    assert c.started_at == expected["started_at"]
    assert c.ended_at == expected["ended_at"]
    assert len(c.turns) == len(expected["turns"])
    for got, exp in zip(c.turns, expected["turns"]):
        assert got.index == exp["index"]
        assert got.role == exp["role"]
        assert got.text == exp["text"]
        assert got.timestamp == exp["timestamp"]


def test_chunk_indices_are_per_chunk_zero_based():
    chunks = list(ClaudeCodeAdapter(FIX, idle_minutes=30).iter_chunks())
    for c in chunks:
        assert [t.index for t in c.turns] == list(range(len(c.turns)))


def test_chunk_ids_deterministic_across_runs():
    ids1 = [c.chunk_id for c in ClaudeCodeAdapter(FIX, idle_minutes=30).iter_chunks()]
    ids2 = [c.chunk_id for c in ClaudeCodeAdapter(FIX, idle_minutes=30).iter_chunks()]
    assert ids1 == ids2
    assert ids1 == [
        "claude_code_sess-synthetic-001_000",
        "claude_code_sess-synthetic-001_001",
    ]


def test_burst_split_at_exactly_idle_threshold(tmp_path):
    # A gap of exactly idle_minutes is NOT a boundary (split is on > only).
    f = tmp_path / "s.jsonl"
    _write(
        f,
        [
            _cc_line("user", "2026-05-20T10:00:00Z", "a"),
            _cc_line("assistant", "2026-05-20T10:30:00Z", "b"),  # exactly 30 min
            _cc_line("user", "2026-05-20T11:00:01Z", "c"),  # > 30 min after prev
        ],
    )
    chunks = list(ClaudeCodeAdapter(f, idle_minutes=30).iter_chunks())
    assert [len(c.turns) for c in chunks] == [2, 1]


def test_chunk_max_turns_subsplits_long_burst(tmp_path):
    f = tmp_path / "s.jsonl"
    lines = [
        _cc_line("user", f"2026-05-20T10:{i:02d}:00Z", f"t{i}", uuid=str(i))
        for i in range(10)
    ]
    _write(f, lines)
    chunks = list(
        ClaudeCodeAdapter(f, idle_minutes=30, chunk_max_turns=4).iter_chunks()
    )
    assert [len(c.turns) for c in chunks] == [4, 4, 2]
    assert [c.chunk_id.endswith(s) for c, s in zip(chunks, ("000", "001", "002"))]


def test_robust_to_malformed_and_junk_lines(tmp_path):
    f = tmp_path / "s.jsonl"
    f.write_text(
        "\n".join(
            [
                _cc_line("user", "2026-05-20T10:00:00Z", "real-a"),
                "{ not valid json",
                "",
                json.dumps({"type": "summary", "summary": "skip me"}),
                json.dumps({"type": "user"}),  # no message / timestamp
                "garbage line not even json",
                _cc_line("assistant", "2026-05-20T10:01:00Z", "real-b"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    chunks = list(ClaudeCodeAdapter(f, idle_minutes=30).iter_chunks())
    assert len(chunks) == 1
    assert [t.text for t in chunks[0].turns] == ["real-a", "real-b"]


def test_directory_of_multiple_session_files(tmp_path):
    d = tmp_path / "sessions"
    d.mkdir()
    _write(d / "s1.jsonl", [_cc_line("user", "2026-05-20T10:00:00Z", "one", "S1")])
    _write(d / "s2.jsonl", [_cc_line("user", "2026-05-20T10:00:00Z", "two", "S2")])
    chunks = list(ClaudeCodeAdapter(d, idle_minutes=30).iter_chunks())
    assert {c.session_id for c in chunks} == {"S1", "S2"}
    assert len(chunks) == 2


def test_session_spanning_two_files_is_merged(tmp_path):
    d = tmp_path / "sessions"
    d.mkdir()
    _write(d / "a.jsonl", [_cc_line("user", "2026-05-20T10:00:00Z", "first", "S")])
    _write(
        d / "b.jsonl", [_cc_line("assistant", "2026-05-20T10:02:00Z", "second", "S")]
    )
    chunks = list(ClaudeCodeAdapter(d, idle_minutes=30).iter_chunks())
    assert len(chunks) == 1
    assert [t.text for t in chunks[0].turns] == ["first", "second"]


def test_assistant_content_as_plain_string_tolerated(tmp_path):
    f = tmp_path / "s.jsonl"
    rec = {
        "type": "assistant",
        "timestamp": "2026-05-20T10:00:00Z",
        "sessionId": "S",
        "uuid": "u",
        "message": {"role": "assistant", "content": "plain string reply"},
    }
    f.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    chunks = list(ClaudeCodeAdapter(f, idle_minutes=30).iter_chunks())
    assert chunks[0].turns[0].text == "plain string reply"


def test_assistant_multiblock_content_concatenated(tmp_path):
    f = tmp_path / "s.jsonl"
    rec = {
        "type": "assistant",
        "timestamp": "2026-05-20T10:00:00Z",
        "sessionId": "S",
        "uuid": "u",
        "message": {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "part-1 "},
                {"type": "tool_use", "name": "x"},  # non-text block ignored
                {"type": "text", "text": "part-2"},
            ],
        },
    }
    f.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    chunks = list(ClaudeCodeAdapter(f, idle_minutes=30).iter_chunks())
    assert chunks[0].turns[0].text == "part-1 part-2"


def test_empty_file_yields_no_chunks(tmp_path):
    f = tmp_path / "empty.jsonl"
    f.write_text("", encoding="utf-8")
    assert list(ClaudeCodeAdapter(f, idle_minutes=30).iter_chunks()) == []


def test_missing_path_yields_no_chunks(tmp_path):
    assert list(ClaudeCodeAdapter(tmp_path / "nope.jsonl").iter_chunks()) == []


def test_split_bursts_helper_empty():
    assert split_bursts([], idle_minutes=30) == []


def test_split_bursts_helper_single_turn():
    t = [Turn(0, "user", "x", "2026-05-20T10:00:00Z")]
    assert split_bursts(t, idle_minutes=30) == [t]
