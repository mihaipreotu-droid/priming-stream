"""R1 — endpoint contract for ``POST /v1/reload`` (v0.7-x-daemon-reload spec §2).

Strategy mirrors :mod:`test_daemon_server`: monkeypatch server module-level
state to stubs and replace ``RecordsVecIndex`` with a fake constructor so
no fastembed/ChromaDB init runs. The HTTP surface is hit through a real
in-process ``ThreadingHTTPServer``.
"""
from __future__ import annotations

import http.client
import json
import threading
import time
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from priming_stream.daemon import lifecycle, server


# ---- fixtures -----------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_daemon_dir(monkeypatch, tmp_path):
    monkeypatch.setenv(lifecycle.DAEMON_DIR_ENV, str(tmp_path / "daemon"))


class _StubVecIndex:
    """Counts ``.count()`` calls and exposes a settable count.

    Each call to ``search()`` is recorded so tests can verify warmup ran.
    """

    instances: list["_StubVecIndex"] = []

    def __init__(self, persist_dir=None, model_name=None, *, count: int = 5):
        self.persist_dir = persist_dir
        self.model_name = model_name
        self._count = count
        self.search_calls: list[tuple[str, int]] = []
        _StubVecIndex.instances.append(self)

    def count(self) -> int:
        return self._count

    def search(self, query: str, k: int):
        self.search_calls.append((query, k))
        return []


@pytest.fixture
def stub_state(monkeypatch, tmp_path):
    """Install stub state on the server module so the handler is exercisable
    without paying fastembed/Chroma cost."""
    _StubVecIndex.instances.clear()
    initial = _StubVecIndex(count=7)
    monkeypatch.setattr(server, "_vec_index", initial)
    monkeypatch.setattr(server, "_repo", object())
    monkeypatch.setattr(server, "_bridge_cfg", object())
    monkeypatch.setattr(server, "_model_name", "stub-model")
    monkeypatch.setattr(server, "_started_at", "2026-05-27T00:00:00Z")
    monkeypatch.setattr(server, "_started_monotonic", time.monotonic())
    monkeypatch.setattr(server, "_model_warm_ok", True)
    # Replace RecordsVecIndex constructor used by _handle_reload with the
    # stub class. The constructor returns a fresh _StubVecIndex with the
    # incremented count so tests can verify the swap happened.
    monkeypatch.setattr(
        server, "RecordsVecIndex",
        lambda persist_dir, model_name: _StubVecIndex(
            persist_dir=persist_dir, model_name=model_name, count=10,
        ),
    )
    # Point the reload handler's path resolution at a tmp dir so its SQLite
    # connect succeeds without a real canonical storage/graph.db (hermetic on
    # a fresh clone). RecordsVecIndex is stubbed, so vec_index_dir is unused.
    monkeypatch.setattr(
        server, "resolve_paths",
        lambda cfg: SimpleNamespace(
            graph_db=tmp_path / "graph.db", vec_index_dir=tmp_path / "vec",
        ),
    )
    # Quiet logging into tmp dir.
    import logging as _logging
    monkeypatch.setattr(server._logger, "handlers", [_logging.NullHandler()])
    return initial


@pytest.fixture
def live_server(stub_state):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), server._Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=2.0)


def _post(srv, path: str, body: dict | None) -> tuple[int, dict]:
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_port, timeout=5)
    payload = json.dumps(body or {}).encode("utf-8") if body is not None else b""
    headers = {"Content-Type": "application/json"} if body is not None else {}
    conn.request("POST", path, body=payload, headers=headers)
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    try:
        parsed = json.loads(data.decode("utf-8"))
    except Exception:
        parsed = {"_raw": data}
    return resp.status, parsed


def _post_raw(srv, path: str, raw_body: bytes) -> tuple[int, dict]:
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_port, timeout=5)
    conn.request(
        "POST", path, body=raw_body,
        headers={"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    try:
        parsed = json.loads(data.decode("utf-8"))
    except Exception:
        parsed = {"_raw": data}
    return resp.status, parsed


def _get(srv, path: str) -> tuple[int, dict]:
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_port, timeout=5)
    conn.request("GET", path)
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    try:
        parsed = json.loads(data.decode("utf-8"))
    except Exception:
        parsed = {"_raw": data}
    return resp.status, parsed


# ---- R1.a — response shape ---------------------------------------------


def test_r1a_reload_returns_200_with_required_keys(live_server):
    status, body = _post(live_server, "/v1/reload", {})
    assert status == 200
    assert set(body.keys()) == {
        "status", "reload_ms", "records_before",
        "records_after", "daemon_version",
    }
    assert body["status"] == "ok"
    assert body["records_before"] == 7  # initial stub
    assert body["records_after"] == 10  # new stub
    assert isinstance(body["reload_ms"], (int, float))
    assert body["reload_ms"] >= 0
    assert body["daemon_version"] == server.DAEMON_VERSION


# ---- R1.b — atomic swap replaces the global ----------------------------


def test_r1b_reload_replaces_vec_index_singleton(live_server, stub_state):
    initial_id = id(server._vec_index)
    status, _ = _post(live_server, "/v1/reload", {})
    assert status == 200
    new_id = id(server._vec_index)
    assert new_id != initial_id
    # Pre-existing reference (`stub_state`) still holds the old object.
    assert id(stub_state) == initial_id


# ---- R1.c — /v1/health reflects new count ------------------------------


def test_r1c_health_reflects_new_records_count_after_reload(live_server):
    status, body = _get(live_server, "/v1/health")
    assert status == 200
    assert body["records_count"] == 7

    status, _ = _post(live_server, "/v1/reload", {})
    assert status == 200

    status, body = _get(live_server, "/v1/health")
    assert status == 200
    assert body["records_count"] == 10


# ---- R1.d — extra keys tolerated ---------------------------------------


def test_r1d_reload_tolerates_extra_body_keys(live_server):
    status, body = _post(
        live_server, "/v1/reload",
        {"reason": "test", "foo": 123, "nested": {"x": [1, 2]}},
    )
    assert status == 200
    assert body["status"] == "ok"


# ---- R1.e — empty / {} body both work ----------------------------------


def test_r1e_reload_accepts_empty_body(live_server):
    # No body at all: send raw empty bytes (no Content-Length).
    conn = http.client.HTTPConnection(
        "127.0.0.1", live_server.server_port, timeout=5,
    )
    conn.request("POST", "/v1/reload", body=b"")
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    assert resp.status == 200
    body = json.loads(raw.decode("utf-8"))
    assert body["status"] == "ok"


def test_r1e_reload_accepts_empty_json_object(live_server):
    status, body = _post(live_server, "/v1/reload", {})
    assert status == 200
    assert body["status"] == "ok"


# ---- malformed body still triggers reload (body unused) ----------------


def test_reload_tolerates_malformed_json_body(live_server):
    status, body = _post_raw(live_server, "/v1/reload", b"{not valid json")
    # Body is unused by reload — malformed JSON must not become a 500.
    assert status == 200
    assert body["status"] == "ok"


# ---- warmup runs on the new index --------------------------------------


def test_reload_warms_new_index(live_server, stub_state):
    _post(live_server, "/v1/reload", {})
    # Two _StubVecIndex have been created: the initial one in the fixture
    # and one inside _handle_reload. The new one should have search called.
    assert len(_StubVecIndex.instances) >= 2
    new_index = _StubVecIndex.instances[-1]
    assert new_index.search_calls == [("warmup", 1)]


# ---- model_loaded honesty across reload --------------------------------


def test_reload_warmup_failure_marks_model_loaded_false(
    live_server, monkeypatch, stub_state,
):
    """If warmup search raises, swap still happens but ``_model_warm_ok``
    drops to False — matches the bridge-daemon M-2 fix carried into reload."""
    class _PoisonIndex(_StubVecIndex):
        def search(self, query: str, k: int):
            raise RuntimeError("poison")

    monkeypatch.setattr(
        server, "RecordsVecIndex",
        lambda persist_dir, model_name: _PoisonIndex(
            persist_dir=persist_dir, model_name=model_name, count=9,
        ),
    )
    status, body = _post(live_server, "/v1/reload", {})
    assert status == 200
    assert body["records_after"] == 9
    # Health should now report model_loaded false.
    status, h = _get(live_server, "/v1/health")
    assert status == 200
    assert h["model_loaded"] is False
    assert h["records_count"] == 9


# ---- cross-process visibility fix: reload clears the chroma system cache ----


def test_reload_clears_chroma_system_cache_before_new_index(
    live_server, monkeypatch, stub_state,
):
    """``_handle_reload`` must clear ChromaDB's per-path System cache before
    constructing the new index, otherwise the fresh client reuses a stale
    segment reader and never sees records written by the separate
    sleep-finalize process. Spy that the helper is invoked on every reload.
    (Real cross-process behavior is covered in
    test_daemon_reload_cross_process.py.)"""
    calls = {"n": 0}
    monkeypatch.setattr(
        server, "_clear_chroma_system_cache",
        lambda: calls.__setitem__("n", calls["n"] + 1),
    )
    status, _ = _post(live_server, "/v1/reload", {})
    assert status == 200
    assert calls["n"] == 1, "reload did not clear the chroma system cache"
