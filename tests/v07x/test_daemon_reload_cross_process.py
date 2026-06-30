"""Regression: /v1/reload must surface records written by a SEPARATE process.

This is the test that was MISSING when daemon-reload shipped. The original
R1-R5 suite stubbed ``RecordsVecIndex`` (monkeypatched constructor), so it
never exercised real ChromaDB cross-process behavior — and that is exactly
where the bug hid: ChromaDB caches its System (segment readers) per persist-
path within a process, so a fresh ``RecordsVecIndex`` on the same path reused
a stale reader and served old/empty query results despite a correct count().
``sleep-finalize`` writes the new records in its OWN process; the daemon must
see them after reload.

These tests use real ChromaDB (a hard dependency) with a deterministic
SHA-based fake embedder, so they need no fastembed download and run fast.
They are intentionally UNCONDITIONAL — gating them behind an env var is how
this class of bug regresses unnoticed.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

from priming_stream.daemon import server
from priming_stream.integrations.vec_index import RecordsVecIndex

_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class _FakeEmbedder:
    """Deterministic 32-dim SHA embedder — identical across processes, so a
    record written by the writer subprocess and a query embedded in the test
    process land in the same vector space. No model load."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def embed(self, texts):
        import numpy as np
        for t in texts:
            digest = hashlib.sha256(t.encode("utf-8")).digest()
            arr = np.frombuffer(digest, dtype=np.uint8).astype("float32")
            norm = np.linalg.norm(arr) or 1.0
            yield arr / norm


# A standalone writer run as a separate OS process: opens the SAME persist
# dir via RecordsVecIndex and adds a record through the real add path, with
# the same deterministic embedder inlined.
_WRITER = (
    "import sys, hashlib, numpy as np\n"
    "from priming_stream.integrations.vec_index import RecordsVecIndex\n"
    "class FE:\n"
    "    def __init__(self, m): self.m = m\n"
    "    def embed(self, texts):\n"
    "        for t in texts:\n"
    "            d = hashlib.sha256(t.encode('utf-8')).digest()\n"
    "            a = np.frombuffer(d, dtype=np.uint8).astype('float32')\n"
    "            yield a / (np.linalg.norm(a) or 1.0)\n"
    "idx = RecordsVecIndex(__import__('pathlib').Path(sys.argv[1]), sys.argv[2])\n"
    "idx._model = FE(sys.argv[2])\n"
    "idx.add_record(sys.argv[3], sys.argv[4])\n"
    "print('writer count', idx.count())\n"
)


def _write_in_subprocess(persist_dir: Path, rec_id: str, summary: str) -> str:
    r = subprocess.run(
        [sys.executable, "-c", _WRITER, str(persist_dir), _MODEL, rec_id, summary],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, f"writer failed: {r.stderr[:500]}"
    return r.stdout.strip()


def test_reload_sees_cross_process_writes(tmp_path):
    """The core contract: after a separate process writes a record, the
    daemon's reload path (clear cache + new RecordsVecIndex) finds it via
    search — not just in count()."""
    persist = tmp_path / "chroma"

    # 1) daemon's initial index opens the collection (with one record) and
    #    queries — forcing the HNSW segment reader to initialize.
    idx1 = RecordsVecIndex(persist, _MODEL)
    idx1._model = _FakeEmbedder(_MODEL)
    idx1.add_record("rec_alpha", "alpha topic one")
    assert any(h.record_id == "rec_alpha" for h in idx1.search("alpha topic one", k=5))

    # 2) a SEPARATE process writes a new record (mirrors sleep-finalize).
    _write_in_subprocess(persist, "rec_beta", "beta topic two")

    # 3) reload path: clear the ChromaDB system cache, build a fresh index.
    server._clear_chroma_system_cache()
    idx2 = RecordsVecIndex(persist, _MODEL)
    idx2._model = _FakeEmbedder(_MODEL)

    # 4) the reloaded index must SEE the cross-process record via search,
    #    and count must reflect both records.
    assert idx2.count() == 2
    hits = idx2.search("beta topic two", k=5)
    assert any(h.record_id == "rec_beta" for h in hits), (
        "reload did not surface the cross-process record — the bug is back "
        f"(got {[h.record_id for h in hits]})"
    )


def test_reload_cache_clear_preserves_old_inflight_index(tmp_path):
    """In-flight safety: clearing the system cache for a fresh client must
    NOT break the old index that in-flight /v1/spread requests still hold.
    The atomic-swap invariant depends on this."""
    persist = tmp_path / "chroma"

    idx_old = RecordsVecIndex(persist, _MODEL)
    idx_old._model = _FakeEmbedder(_MODEL)
    idx_old.add_record("rec_alpha", "alpha topic one")
    assert idx_old.search("alpha topic one", k=5)

    _write_in_subprocess(persist, "rec_beta", "beta topic two")

    server._clear_chroma_system_cache()
    idx_new = RecordsVecIndex(persist, _MODEL)
    idx_new._model = _FakeEmbedder(_MODEL)
    _ = idx_new.count()

    # old index (in-flight) still queries fine after the clear + new client.
    old_hits = idx_old.search("alpha topic one", k=5)
    assert any(h.record_id == "rec_alpha" for h in old_hits), (
        "clearing the system cache broke the old in-flight index"
    )
