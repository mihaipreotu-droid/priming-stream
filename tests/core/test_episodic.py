"""Unit — episodic store: events, multi-chunk round-trip, processed tracking."""
import json

from priming_stream.core.episodic import EpisodicStore
from priming_stream.core.models import Chunk, Turn


def _chunk(cid: str) -> Chunk:
    return Chunk(
        chunk_id=cid, source_client="claude_code", session_id="s1",
        started_at="2026-05-20T10:00:00Z", ended_at="2026-05-20T10:05:00Z",
        turns=[Turn(0, "user", "q " + cid, "2026-05-20T10:00:00Z")])


def test_init_creates_dir(tmp_path):
    target = tmp_path / "nested" / "episodic"
    EpisodicStore(target)
    assert target.is_dir()


def test_append_event(tmp_path):
    store = EpisodicStore(tmp_path)
    store.append_event({"kind": "session_start", "n": 1})
    store.append_event({"kind": "session_end", "n": 2})
    lines = (tmp_path / "live_events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["kind"] == "session_start"


def test_multi_chunk_roundtrip(tmp_path):
    store = EpisodicStore(tmp_path)
    for cid in ("c_000", "c_001", "c_002"):
        store.write_chunk(_chunk(cid))
    got = [c.chunk_id for c in store.iter_chunks()]
    assert got == ["c_000", "c_001", "c_002"]
    assert list(store.iter_chunks())[1].turns[0].text == "q c_001"


def test_unprocessed_filtering(tmp_path):
    store = EpisodicStore(tmp_path)
    for cid in ("c_000", "c_001", "c_002"):
        store.write_chunk(_chunk(cid))
    assert len(list(store.iter_unprocessed_chunks())) == 3
    store.mark_processed("c_001", cycle_id=7)
    remaining = [c.chunk_id for c in store.iter_unprocessed_chunks()]
    assert remaining == ["c_000", "c_002"]


def test_unicode_preserved(tmp_path):
    store = EpisodicStore(tmp_path)
    ch = Chunk(
        chunk_id="c_u", source_client="claude_code", session_id="s1",
        started_at="t", ended_at="t",
        turns=[Turn(0, "user", "diacritice: ă î ț â ș", "t")])
    store.write_chunk(ch)
    assert list(store.iter_chunks())[0].turns[0].text == "diacritice: ă î ț â ș"


# -- M1: re-ingestion must be idempotent on chunk_id ----------------------


def test_write_chunk_is_idempotent_on_chunk_id(tmp_path):
    store = EpisodicStore(tmp_path)
    store.write_chunk(_chunk("c_dup"))
    store.write_chunk(_chunk("c_dup"))  # re-ingest same chunk
    chunks = list(store.iter_chunks())
    assert [c.chunk_id for c in chunks] == ["c_dup"]
    lines = (tmp_path / "chunks.jsonl").read_text(
        encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_reingestion_leaves_first_run_unprocessed_set(tmp_path):
    """Ingesting the same set twice yields exactly the first-run chunks."""
    store = EpisodicStore(tmp_path)
    for cid in ("c_000", "c_001", "c_002"):
        store.write_chunk(_chunk(cid))
    first = [c.chunk_id for c in store.iter_unprocessed_chunks()]
    # second ingestion of the identical transcript
    for cid in ("c_000", "c_001", "c_002"):
        store.write_chunk(_chunk(cid))
    second = [c.chunk_id for c in store.iter_unprocessed_chunks()]
    assert first == second == ["c_000", "c_001", "c_002"]


def test_iter_chunks_dedupes_pre_existing_duplicates(tmp_path):
    """Even if duplicates leaked into the file, iteration yields each once."""
    store = EpisodicStore(tmp_path)
    # bypass write_chunk to simulate a pre-existing corrupted file
    store._append(store.chunks_path, {"chunk_id": "c_x", "source_client": "cc",
                  "session_id": "s", "started_at": "t", "ended_at": "t",
                  "turns": []})
    store._append(store.chunks_path, {"chunk_id": "c_x", "source_client": "cc",
                  "session_id": "s", "started_at": "t", "ended_at": "t",
                  "turns": []})
    assert [c.chunk_id for c in store.iter_chunks()] == ["c_x"]
