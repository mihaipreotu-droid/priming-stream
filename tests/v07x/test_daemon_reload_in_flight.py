"""R2 — atomic swap / in-flight request semantics.

R2.a uses a slow ``spread`` to demonstrate that a request started before
reload completes against the old index. It runs in-process against a
``ThreadingHTTPServer``; no real daemon subprocess needed, but it is gated
behind ``@pytest.mark.daemon`` per the panel spec (real concurrent HTTP
traffic on the loopback is the binding test).

R2.b verifies that ``/v1/spread`` started after a reload sees the new index.
This is the more practical assertion and runs unconditionally.
"""
from __future__ import annotations

import http.client
import json
import os
import threading
import time
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from priming_stream.bridge.types import PrimingResult, ScoredRecord
from priming_stream.core.models import Record
from priming_stream.daemon import lifecycle, server


@pytest.fixture(autouse=True)
def _isolated_daemon_dir(monkeypatch, tmp_path):
    monkeypatch.setenv(lifecycle.DAEMON_DIR_ENV, str(tmp_path / "daemon"))


class _StubVecIndex:
    instances: list["_StubVecIndex"] = []

    def __init__(self, persist_dir=None, model_name=None, *, count: int = 5):
        self._count = count
        _StubVecIndex.instances.append(self)

    def count(self) -> int:
        return self._count

    def search(self, query: str, k: int):
        return []

    def embed_texts(self, texts):
        # canary-friendly stub: related pair collinear, control orthogonal
        # (the reload path now runs the embedder-identity canary).
        vecs = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        return vecs[: len(texts)]


def _install_stub_state(monkeypatch, *, initial_count: int, reload_count: int):
    _StubVecIndex.instances.clear()
    initial = _StubVecIndex(count=initial_count)
    monkeypatch.setattr(server, "_vec_index", initial)
    monkeypatch.setattr(server, "_repo", object())
    monkeypatch.setattr(server, "_bridge_cfg", object())
    monkeypatch.setattr(server, "_model_name", "stub-model")
    monkeypatch.setattr(server, "_started_at", "2026-05-27T00:00:00Z")
    monkeypatch.setattr(server, "_started_monotonic", time.monotonic())
    monkeypatch.setattr(server, "_model_warm_ok", True)
    monkeypatch.setattr(
        server, "RecordsVecIndex",
        lambda persist_dir, model_name: _StubVecIndex(
            persist_dir=persist_dir, model_name=model_name,
            count=reload_count,
        ),
    )
    import logging as _logging
    monkeypatch.setattr(server._logger, "handlers", [_logging.NullHandler()])
    return initial


@pytest.fixture
def live_server(monkeypatch, tmp_path):
    _install_stub_state(monkeypatch, initial_count=3, reload_count=11)
    # Hermetic: point the reload handler's SQLite connect at a tmp graph.db so
    # it doesn't require a real canonical storage/graph.db on a fresh clone.
    monkeypatch.setattr(
        server, "resolve_paths",
        lambda cfg: SimpleNamespace(
            graph_db=tmp_path / "graph.db", vec_index_dir=tmp_path / "vec",
        ),
    )
    srv = ThreadingHTTPServer(("127.0.0.1", 0), server._Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=2.0)


def _post(srv, path: str, body: dict) -> tuple[int, dict]:
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_port, timeout=10)
    payload = json.dumps(body).encode("utf-8")
    conn.request(
        "POST", path, body=payload,
        headers={"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, json.loads(data.decode("utf-8"))


# ---- R2.b — new spread after reload sees the new index -----------------


def _priming_with_count(vec_index) -> PrimingResult:
    """One fake semantic record whose summary reports ``vec_index.count()`` —
    lets the test prove WHICH index reference build_priming saw."""
    return PrimingResult(
        semantic=[
            ScoredRecord(
                record=Record(
                    id="rec_0000aaaa",
                    source_uri="qmd://corpus/x.md",
                    anchor_offset_start=0,
                    anchor_offset_end=10,
                    summary=f"count={vec_index.count()}",
                    created_at="2026-05-27T00:00:00Z",
                ),
                score=1.0,
            ),
        ],
        lexical=[],
    )


def test_reload_refreshes_bridge_cfg_and_model_name(live_server):
    """/v1/reload must pick up the CURRENT config —
    [bridge] knob edits apply on the nightly reload, not only on restart;
    /v1/health's model_name follows the live config too."""
    old_cfg = server._bridge_cfg
    status, _ = _post(live_server, "/v1/reload", {})
    assert status == 200
    assert server._bridge_cfg is not old_cfg
    assert server._model_name != "stub-model"


def test_spread_refuses_when_canary_failed(live_server, monkeypatch):
    """Canary gate: an unverified embedder must refuse spread
    (503 → hook client None → lexical fallback), never serve garbage."""
    monkeypatch.setattr(server, "_canary_ok", False)
    status, body = _post(live_server, "/v1/spread", {"prompt_text": "x"})
    assert status == 503
    assert "canary" in body["error"]


def test_run_canary_separation():
    class _Good:
        def embed_texts(self, texts):
            return [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]][: len(texts)]

    class _Flat:  # weightless/wrong artifact: no separation
        def embed_texts(self, texts):
            return [[1.0, 0.0]] * len(texts)

    assert server._run_canary(_Good()) is True
    assert server._run_canary(_Flat()) is False


def test_r2b_new_spread_after_reload_sees_new_index(live_server, monkeypatch):
    """build_priming is mocked to introspect the live ``_vec_index`` reference
    (passed as the ``vec_index`` kwarg) and report its count as a single fake
    semantic record. Before reload the count is 3 (initial); after reload the
    atomic swap should make build_priming see count=11 (the new stub).
    """
    def _fake_build_priming(prompt, prev, *, vec_index, repo, conn, cfg,
                            now=None, exclude_recent_ids=frozenset(),
                            turn_features=None):
        return _priming_with_count(vec_index)
    monkeypatch.setattr(server, "build_priming", _fake_build_priming)

    status, body = _post(live_server, "/v1/spread", {"prompt_text": "x"})
    assert status == 200
    assert body["semantic"][0]["summary"] == "count=3"

    status, _ = _post(live_server, "/v1/reload", {})
    assert status == 200

    status, body = _post(live_server, "/v1/spread", {"prompt_text": "x"})
    assert status == 200
    assert body["semantic"][0]["summary"] == "count=11"


# ---- R2.a — in-flight spread completes on old state (gated) -----------


@pytest.mark.daemon
@pytest.mark.skipif(
    os.environ.get("RUN_DAEMON_TESTS") != "1",
    reason="set RUN_DAEMON_TESTS=1 to run concurrent in-flight HTTP test",
)
def test_r2a_inflight_spread_completes_on_old_index(live_server, monkeypatch):
    """Start a /v1/spread that sleeps ~500ms inside the spread call. Fire
    /v1/reload while it's blocked. Assert spread returns 200 with a summary
    pointing at the OLD index (count=3), not the new one (count=11).

    Gated because real concurrent HTTP on loopback + ThreadingHTTPServer
    is closer to integration than unit.
    """
    def _slow_build_priming(prompt, prev, *, vec_index, repo, conn, cfg,
                            now=None):
        captured_count = vec_index.count()
        time.sleep(0.5)
        return PrimingResult(
            semantic=[
                ScoredRecord(
                    record=Record(
                        id="rec_0000aaaa",
                        source_uri="qmd://corpus/x.md",
                        anchor_offset_start=0,
                        anchor_offset_end=10,
                        summary=f"count={captured_count}",
                        created_at="2026-05-27T00:00:00Z",
                    ),
                    score=1.0,
                ),
            ],
            lexical=[],
        )
    monkeypatch.setattr(server, "build_priming", _slow_build_priming)

    spread_result: dict = {}

    def _runner():
        status, body = _post(live_server, "/v1/spread", {"prompt_text": "x"})
        spread_result["status"] = status
        spread_result["body"] = body

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    # Let the spread start and hit the sleep.
    time.sleep(0.1)

    # Fire reload — this should complete promptly.
    status, body = _post(live_server, "/v1/reload", {})
    assert status == 200
    assert body["records_after"] == 11

    # Wait for the spread to finish.
    t.join(timeout=5.0)
    assert not t.is_alive(), "spread did not complete"

    assert spread_result["status"] == 200
    # build_priming captured count at entry — count=3 (the OLD index).
    assert spread_result["body"]["semantic"][0]["summary"] == "count=3"
