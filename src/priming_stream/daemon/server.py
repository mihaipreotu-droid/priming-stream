"""Resident bridge daemon — HTTP server holding warm embedding model +
ChromaDB client + bridge spread logic across CC sessions.

Started detached by the hook autostart path or via ``prime daemon start``.
Listens on 127.0.0.1, OS-assigned port. Writes the endpoint discovery file
(``daemon.json``) on startup; removes it on shutdown.

Stdout discipline (binding, spec §5 row 4): the daemon NEVER writes to
stdout. All logs go to ``daemon.log`` via :class:`RotatingFileHandler`.
The detached subprocess has its stdio redirected to DEVNULL by
``lifecycle.autostart_daemon``; we additionally avoid library prints by
configuring ``_logger.propagate = False`` and never installing a
``StreamHandler``.

Exposes:

* ``POST /v1/spread`` — accepts ``{prompt_text, prev_assistant_text?,
  session_id?}``, calls :func:`priming_stream.bridge.working_set.build_priming`,
  returns the two priming buckets (``semantic`` + ``lexical``) as JSON
  (v0.7-x Component A two-bucket shape).
* ``GET /v1/health`` — uptime, records_count, model_loaded, model_name,
  daemon_version.

Singleton via :func:`priming_stream.daemon.lifecycle.acquire_lock`; a second daemon
attempting startup fails fast and exits cleanly. Stops on SIGINT (Ctrl-C in
foreground) or SIGTERM; cleanup removes the endpoint file and releases the
lockfile.

This module is the heavyweight boundary (spec §5 row 3): it may import
``priming_stream.integrations.vec_index`` and ``priming_stream.bridge.spreading``. The
hook/client/fallback path remains stdlib-only.
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from priming_stream.bridge.working_set import build_priming, priming_items
from priming_stream.core.config import load_config
from priming_stream.core.db import connect
from priming_stream.core.graph_repo import GraphRepo
from priming_stream.core.paths import resolve_paths
from priming_stream.daemon import lifecycle
from priming_stream.integrations.vec_index import RecordsVecIndex

DAEMON_VERSION = "v0.7-x-bridge-daemon"

# Module-level singletons populated by ``run_server``. The handler reads
# these directly; tests override them via monkeypatch to exercise the HTTP
# surface without paying the fastembed/Chroma init cost.
_started_at: str | None = None
_started_monotonic: float | None = None
_vec_index: RecordsVecIndex | None = None
_repo: GraphRepo | None = None
_conn = None  # sqlite3.Connection
_bridge_cfg = None  # BridgeConfig
_model_name: str = ""  # mirrored from cfg.vec_index.model_name; see /v1/health note
# True only when warmup succeeded. A poisoned model still leaves
# ``_vec_index`` set (so /v1/health can report records_count etc.),
# but ``model_loaded`` must reflect actual usability.
_model_warm_ok: bool = False

_logger = logging.getLogger("priming_stream.daemon")


def _setup_logging() -> None:
    """Configure the daemon's rotating file logger.

    10MB × 3 backups at :func:`lifecycle.log_path`. Honors
    ``PRIMING_STREAM_DAEMON_LOG_LEVEL`` (default ``INFO``). Removes any
    pre-existing handlers and disables propagation so a stray root
    ``StreamHandler`` cannot leak to stdout/stderr.
    """
    level_name = os.environ.get("PRIMING_STREAM_DAEMON_LOG_LEVEL", "INFO")
    level = getattr(logging, level_name.upper(), logging.INFO)
    _logger.setLevel(level)
    for h in list(_logger.handlers):
        _logger.removeHandler(h)
    handler = logging.handlers.RotatingFileHandler(
        lifecycle.log_path(),
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    _logger.addHandler(handler)
    _logger.propagate = False


def _init_state() -> None:
    """Load config, open ChromaDB + SQLite, warm the embedder.

    The warmup call (``vec_index.search("warmup", k=1)``) forces fastembed
    to initialize before the first ``/v1/spread`` request, so users don't
    pay the ~2-3s model-load cost on their first prompt after autostart.
    """
    global _started_at, _started_monotonic, _vec_index, _repo, _conn
    global _bridge_cfg, _model_name, _model_warm_ok
    cfg = load_config()
    paths = resolve_paths(cfg)
    _started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _started_monotonic = time.monotonic()
    _model_name = cfg.vec_index.model_name
    _vec_index = RecordsVecIndex(paths.vec_index_dir, cfg.vec_index.model_name)
    _model_warm_ok = False
    try:
        _vec_index.search("warmup", k=1)
        _model_warm_ok = True
    except Exception as exc:  # noqa: BLE001
        _logger.warning("warmup search failed: %s", exc)
    # ThreadingHTTPServer dispatches each request on a worker thread, but
    # the connection here is opened on the main thread at startup. SQLite
    # by default refuses cross-thread access. Pass ``check_same_thread=False``
    # — safe because the daemon's bridge path is read-only (records table
    # via GraphRepo.get_record), WAL is enabled, and concurrent SELECTs
    # don't need explicit serialization. If the daemon ever grows a write
    # path, add a threading.Lock around it.
    import sqlite3 as _sqlite3
    _conn = _sqlite3.connect(str(paths.graph_db), check_same_thread=False)
    _conn.row_factory = _sqlite3.Row
    _conn.execute("PRAGMA foreign_keys = ON")
    _conn.execute("PRAGMA busy_timeout = 5000")
    _repo = GraphRepo(_conn)
    _bridge_cfg = cfg.bridge
    _logger.info(
        "daemon initialized version=%s records=%d",
        DAEMON_VERSION,
        _vec_index.count(),
    )


def _clear_chroma_system_cache() -> None:
    """Drop ChromaDB's per-path System cache so the next ``PersistentClient``
    re-reads segments from disk.

    ChromaDB caches a ``System`` (SegmentManager + HNSW segment readers) per
    persist-path *within a process* (``SharedSystemClient``). A second
    ``PersistentClient`` on the same path reuses that cached System, whose
    segment readers were initialized at first open and do NOT pick up vectors
    written by a *separate* process — and ``sleep-finalize`` writes the new
    records in its own process. Without this clear, ``/v1/reload`` reports a
    correct ``count()`` (read from SQLite metadata) but ``query`` either errors
    ("Nothing found on disk", if the collection was first opened empty) or
    silently returns only the stale pre-reload records. Confirmed + fix
    verified 2026-05-29.

    Clearing only empties the cache dict; live ``System`` objects already held
    by the old ``_vec_index`` stay alive, so in-flight ``/v1/spread`` requests
    on the old index are unaffected (verified: old client still queries fine
    after a clear + new-client construction).
    """
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
        SharedSystemClient.clear_system_cache()
    except Exception as exc:  # noqa: BLE001 - best effort, never block reload
        _logger.warning("could not clear chroma system cache: %s", exc)


def _scored_to_item(sr, rank: int) -> dict:
    """Serialize one :class:`ScoredRecord` to a per-record response item.

    Delegates to :func:`priming_stream.bridge.working_set._scored_to_item` which is
    the shared conversion used by daemon, MCP, and tests. This local name is
    kept for backward-compatibility (existing tests import it from here).
    """
    from priming_stream.bridge.working_set import _scored_to_item as _shared
    return _shared(sr, rank)


class _Handler(BaseHTTPRequestHandler):
    # BaseHTTPRequestHandler writes its access log to stderr by default;
    # redirect into our file logger to keep stderr silent.
    def log_message(self, format, *args):  # noqa: A002
        _logger.debug("http %s %s", self.address_string(), format % args)

    def do_GET(self):  # noqa: N802
        if self.path == "/v1/health":
            self._handle_health()
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if self.path == "/v1/spread":
            self._handle_spread()
            return
        if self.path == "/v1/reload":
            self._handle_reload()
            return
        self._send_json(404, {"error": "not found"})

    def _handle_health(self) -> None:
        try:
            uptime = time.monotonic() - (_started_monotonic or time.monotonic())
            body = {
                "status": "ok",
                "uptime_s": round(uptime, 1),
                "records_count": _vec_index.count() if _vec_index else 0,
                # Reports warmup success, not just object presence: a
                # poisoned model leaves ``_vec_index`` set but unusable.
                "model_loaded": _model_warm_ok,
                # ``RecordsVecIndex`` exposes ``_model_name`` (private). We
                # mirror the same string into ``_model_name`` at init from
                # ``cfg.vec_index.model_name`` so we don't reach across the
                # encapsulation boundary on the hot path.
                "model_name": _model_name,
                "daemon_version": DAEMON_VERSION,
            }
            self._send_json(200, body)
        except Exception as exc:  # noqa: BLE001
            _logger.exception("health error")
            self._send_json(500, {"error": str(exc)})

    def _handle_spread(self) -> None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length > 0 else b"{}"
            req = json.loads(raw.decode("utf-8") or "{}")
            if not isinstance(req, dict):
                req = {}
            prompt = str(req.get("prompt_text") or "")
            prev = str(req.get("prev_assistant_text") or "")
            session_id = req.get("session_id") or None
            # Item 3.3: ids primed in the last N turns of this session, dropped
            # before each bucket's truncation so freed slots backfill. The hook
            # computes the window (it owns the echo history); the daemon just
            # applies the set. Defensive: tolerate a missing / non-list field.
            raw_recent = req.get("recent_ids")
            recent_ids = frozenset(
                str(r) for r in raw_recent if r
            ) if isinstance(raw_recent, list) else frozenset()
            t0 = time.monotonic()
            priming = build_priming(
                prompt, prev,
                vec_index=_vec_index, repo=_repo, conn=_conn, cfg=_bridge_cfg,
                exclude_recent_ids=recent_ids,
            )
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            sem_items, lex_items = priming_items(priming)
            body = {
                "semantic": sem_items,
                "lexical": lex_items,
                "spread_ms": round(elapsed_ms, 2),
                "daemon_version": DAEMON_VERSION,
            }
            _logger.info(
                "spread session=%s prompt_len=%d semantic=%d lexical=%d ms=%.1f",
                session_id,
                len(prompt),
                len(priming.semantic),
                len(priming.lexical),
                elapsed_ms,
            )
            self._send_json(200, body)
        except Exception as exc:  # noqa: BLE001
            _logger.exception("spread error")
            self._send_json(500, {"error": str(exc)})

    def _handle_reload(self) -> None:
        """POST /v1/reload handler — atomic swap of ``_vec_index``.

        Reload reuses the daemon's already-loaded fastembed model via the
        process-level cache in ``RecordsVecIndex`` (see its module
        docstring), so the new instance attaches the existing ONNX session
        in ~25ms total instead of paying the ~1.5-2.5s session-init cost.
        Atomic-swap pattern means in-flight /v1/spread requests captured
        the old _vec_index reference locally and complete unaffected.

        Cross-process visibility: the new records were written to ChromaDB by
        a *separate* ``sleep-finalize`` process. ChromaDB caches its System
        per persist-path within this process, so a fresh ``RecordsVecIndex``
        would otherwise reuse a stale segment reader and serve old/empty query
        results despite a correct count. ``_clear_chroma_system_cache()`` below
        forces the new client to re-read segments from disk. See that helper.

        Body is ignored — empty body, ``{}``, or extra keys all tolerated
        per spec §4.1 / R1.d / R1.e. Request body parsing is best-effort:
        a malformed JSON body still triggers a reload (the body is unused).
        """
        global _vec_index, _model_warm_ok, _conn, _repo
        # Drain any request body so the client side doesn't see a broken
        # pipe; contents are ignored per spec.
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length > 0:
                self.rfile.read(length)
        except Exception:
            pass
        t0 = time.monotonic()
        old_index = _vec_index
        old_conn = _conn
        records_before = old_index.count() if old_index is not None else 0
        try:
            cfg = load_config()
            paths = resolve_paths(cfg)
            # Force a fresh ChromaDB System so the new client below sees
            # records written by the separate sleep-finalize process. Without
            # this, the reloaded index reports a correct count() but serves
            # stale/empty query results (see _clear_chroma_system_cache).
            _clear_chroma_system_cache()
            new_index = RecordsVecIndex(
                paths.vec_index_dir, cfg.vec_index.model_name,
            )
            # Warm the new index's model handle. A failed warmup is logged
            # but doesn't abort the swap — ``_model_warm_ok`` reflects the
            # actual usability of the new index for /v1/health, matching
            # the bridge-daemon M-2 fix.
            try:
                new_index.search("warmup", k=1)
                warm_ok = True
            except Exception as warm_exc:  # noqa: BLE001
                _logger.warning("reload warmup search failed: %s", warm_exc)
                warm_ok = False
            # Build a fresh SQLite connection + GraphRepo so records added
            # since daemon startup are visible to subsequent /v1/spread calls.
            # We open the new connection BEFORE the atomic swap so we never
            # leave the daemon with a None _conn if the open fails.
            import sqlite3 as _sqlite3
            new_conn = _sqlite3.connect(
                str(paths.graph_db), check_same_thread=False,
            )
            new_conn.row_factory = _sqlite3.Row
            new_conn.execute("PRAGMA foreign_keys = ON")
            new_conn.execute("PRAGMA busy_timeout = 5000")
            new_repo = GraphRepo(new_conn)
            # Atomic single-reference assignments under GIL. In-flight
            # ``_handle_spread`` calls captured the old references locally
            # at entry; they continue against old_index / old_conn unaffected.
            _vec_index = new_index
            _model_warm_ok = warm_ok
            _conn = new_conn
            _repo = new_repo
            # Close old connection AFTER swap — in-flight requests already
            # hold the old reference and will finish normally.
            if old_conn is not None:
                try:
                    old_conn.close()
                except Exception:
                    pass
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            records_after = new_index.count()
            _logger.info(
                "reload ok records_before=%d records_after=%d ms=%.1f",
                records_before, records_after, elapsed_ms,
            )
            self._send_json(200, {
                "status": "ok",
                "reload_ms": round(elapsed_ms, 2),
                "records_before": records_before,
                "records_after": records_after,
                "daemon_version": DAEMON_VERSION,
            })
        except Exception as exc:  # noqa: BLE001 - keep old state on failure
            _logger.exception("reload failed")
            self._send_json(500, {"error": str(exc)})

    def _send_json(self, status: int, body: dict) -> None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def run_server(
    host: str = "127.0.0.1",
    port: int = 0,
    *,
    daemon_version: str = DAEMON_VERSION,  # noqa: ARG001 (kept for spec §4.3 signature)
) -> None:
    """Blocking entry point: acquire lock, init state, serve until signalled.

    Args:
        host: bind host (default ``127.0.0.1`` — never expose externally).
        port: bind port (default ``0`` = OS-assigned free port).
        daemon_version: present for spec §4.3 signature compatibility;
            the constant :data:`DAEMON_VERSION` is the source of truth.

    Behavior on lock contention (another daemon already running): the
    function logs an error and returns cleanly. We never raise to the
    detached parent (which isn't listening) and never write to stdout.
    """
    _setup_logging()
    _logger.info("acquiring daemon lock")
    lock_handle: object | None = None
    try:
        lock_handle = lifecycle.acquire_lock()
    except (BlockingIOError, OSError) as exc:
        _logger.error("another daemon is already running: %s", exc)
        return

    try:
        _init_state()
        server = ThreadingHTTPServer((host, port), _Handler)
        actual_port = server.server_port
        lifecycle.write_endpoint(
            host=host,
            port=actual_port,
            pid=os.getpid(),
            started_at=_started_at or "",
            version=DAEMON_VERSION,
        )
        _logger.info("listening on %s:%d", host, actual_port)

        def _shutdown(signum, _frame):
            _logger.info("received signal %d; shutting down", signum)
            # server.shutdown() blocks until serve_forever returns; call it
            # from a worker thread so the signal handler returns promptly.
            threading.Thread(target=server.shutdown, daemon=True).start()

        signal.signal(signal.SIGINT, _shutdown)
        try:
            signal.signal(signal.SIGTERM, _shutdown)
        except (AttributeError, ValueError):
            # SIGTERM may be unavailable in some Windows contexts.
            pass

        try:
            server.serve_forever()
        finally:
            server.server_close()
    finally:
        lifecycle.remove_endpoint()
        lifecycle.release_lock(lock_handle)
        if _conn is not None:
            try:
                _conn.close()
            except Exception:
                pass
        _logger.info("daemon stopped")


if __name__ == "__main__":
    run_server()
