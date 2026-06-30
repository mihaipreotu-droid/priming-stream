"""``prime daemon start|stop|status|restart`` — bridge daemon control CLI.

Thin wrapper around :mod:`priming_stream.daemon.lifecycle` + (for foreground
``start``) :mod:`priming_stream.daemon.server`. Designed to stay light on the
common paths: ``status`` and ``stop`` only touch the endpoint file and
the network/signal layer — they never import the heavyweight server
module (which would pull fastembed / ChromaDB). The foreground ``start``
branch defers that import until it actually needs to serve.

Subcommands:

* ``start`` — foreground blocking by default (runs
  :func:`priming_stream.daemon.server.run_server`); with ``--background``,
  spawns a detached subprocess via
  :func:`priming_stream.daemon.lifecycle.autostart_daemon` and waits up to
  ~30s for the endpoint file to appear (covers fastembed cold load).
* ``stop`` — read endpoint file, signal the pid (POSIX SIGTERM;
  Windows ``os.kill`` maps SIGTERM to ``TerminateProcess`` — abrupt,
  but the daemon's lifecycle ``finally`` block still removes the
  endpoint file and releases the lock on its own next start), then
  poll for clean shutdown for up to 3s.
* ``status`` — read endpoint file, GET ``/v1/health``; print uptime,
  records_count, model_name, daemon_version.
* ``restart`` — stop + start --background.

Exit codes:
    0 — success (daemon running on ``status``; daemon stopped on
        ``stop``; daemon spawned on ``start --background``).
    1 — failure (daemon not running on ``status`` / ``stop``; autostart
        timeout; endpoint reachable but unhealthy).
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import signal
import socket
import sys
import time

from priming_stream.daemon import lifecycle


# ---------------------------------------------------------------- helpers


_AUTOSTART_TIMEOUT_S = 30.0
"""How long ``start --background`` waits for the endpoint file to appear.
Cold fastembed model load typically completes within 5-10s; 30s is the
spec's documented warmup ceiling."""

_STOP_GRACE_S = 3.0
"""How long ``stop`` waits for the daemon to remove its endpoint file
after the signal is delivered."""


def _print_err(msg: str) -> None:
    print(msg, file=sys.stderr)


# ---------------------------------------------------------------- start


def _cmd_start(args: argparse.Namespace) -> int:
    info = lifecycle.read_endpoint()
    if info and not lifecycle.is_endpoint_stale(info):
        _print_err(
            f"daemon already running on {info.get('host')}:{info.get('port')} "
            f"pid={info.get('pid')}"
        )
        return 1

    if getattr(args, "background", False):
        lifecycle.autostart_daemon()
        deadline = time.monotonic() + _AUTOSTART_TIMEOUT_S
        while time.monotonic() < deadline:
            time.sleep(0.5)
            info = lifecycle.read_endpoint()
            if info and not lifecycle.is_endpoint_stale(info):
                print(
                    f"daemon started on {info.get('host')}:{info.get('port')} "
                    f"pid={info.get('pid')}"
                )
                return 0
        _print_err(
            f"daemon autostart timed out after {_AUTOSTART_TIMEOUT_S:.0f}s "
            "(model load may still be in progress)"
        )
        return 1

    # Foreground: defer the heavyweight import so `daemon status` /
    # `daemon stop` don't load fastembed.
    from priming_stream.daemon.server import run_server
    run_server()
    return 0


# ---------------------------------------------------------------- stop


def _cmd_stop(args: argparse.Namespace) -> int:  # noqa: ARG001
    info = lifecycle.read_endpoint()
    if not info or lifecycle.is_endpoint_stale(info):
        _print_err("daemon not running")
        return 1

    try:
        pid = int(info.get("pid") or 0)
    except (TypeError, ValueError):
        pid = 0
    if pid <= 0:
        _print_err("endpoint file has no valid pid")
        return 1

    # Windows: os.kill(pid, SIGTERM) maps to TerminateProcess (abrupt).
    # POSIX: SIGTERM is delivered cleanly, the daemon's signal handler
    # triggers ThreadingHTTPServer.shutdown(), and the finally block in
    # run_server removes the endpoint file + releases the lock.
    # We accept the abrupt-on-Windows tradeoff: the daemon's lockfile is
    # released by the OS on process death, and a stale endpoint.json is
    # detected by is_endpoint_stale on the next start.
    try:
        os.kill(pid, signal.SIGTERM)
    except (OSError, ValueError) as exc:
        _print_err(f"failed to signal pid {pid}: {exc}")
        return 1

    deadline = time.monotonic() + _STOP_GRACE_S
    while time.monotonic() < deadline:
        time.sleep(0.1)
        info = lifecycle.read_endpoint()
        if not info or lifecycle.is_endpoint_stale(info):
            print(f"daemon stopped (pid={pid})")
            return 0

    # Fallback: the process is gone but didn't clean its endpoint file
    # (e.g., abrupt Windows TerminateProcess). Force-remove so subsequent
    # `status` reports cleanly.
    if not lifecycle.is_pid_alive(pid):
        lifecycle.remove_endpoint()
        print(f"daemon stopped (forced cleanup, pid={pid})")
        return 0
    _print_err(
        f"warning: daemon pid={pid} did not clean up within "
        f"{_STOP_GRACE_S:.0f}s"
    )
    return 1


# ---------------------------------------------------------------- status


def _cmd_status(args: argparse.Namespace) -> int:  # noqa: ARG001
    info = lifecycle.read_endpoint()
    if not info or lifecycle.is_endpoint_stale(info):
        print("daemon not running")
        return 1

    host = str(info.get("host") or "127.0.0.1")
    try:
        port = int(info.get("port") or 0)
    except (TypeError, ValueError):
        port = 0
    if port <= 0:
        _print_err("endpoint file has no valid port")
        return 1

    try:
        conn = http.client.HTTPConnection(host, port, timeout=2.0)
        try:
            conn.request("GET", "/v1/health")
            resp = conn.getresponse()
            raw = resp.read()
            if resp.status != 200:
                _print_err(
                    f"daemon endpoint reachable but /v1/health returned "
                    f"{resp.status}"
                )
                return 1
            body = json.loads(raw.decode("utf-8"))
        finally:
            conn.close()
    except (socket.timeout, ConnectionError, OSError, ValueError) as exc:
        _print_err(f"daemon endpoint present but unreachable: {exc}")
        return 1

    print(f"daemon running on {host}:{port} pid={info.get('pid')}")
    print(f"  version       {body.get('daemon_version')}")
    print(f"  uptime_s      {body.get('uptime_s')}")
    print(f"  records_count {body.get('records_count')}")
    print(f"  model_name    {body.get('model_name')}")
    print(f"  model_loaded  {body.get('model_loaded')}")
    return 0


# ---------------------------------------------------------------- restart


def _cmd_restart(args: argparse.Namespace) -> int:
    # stop returns 1 when no daemon is running; that's fine for restart —
    # we proceed to start regardless. Hard signal-delivery errors also
    # collapse to 1; we still attempt start, which will surface a clearer
    # error if the system is genuinely broken.
    _cmd_stop(args)
    args.background = True
    return _cmd_start(args)


# ---------------------------------------------------------------- register


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "daemon",
        help="start/stop/status/restart the bridge daemon",
    )
    sub = p.add_subparsers(dest="daemon_cmd", required=True)

    p_start = sub.add_parser("start", help="start the daemon")
    p_start.add_argument(
        "--background", action="store_true",
        help="spawn detached and return (default: foreground blocking)",
    )
    p_start.set_defaults(func=_cmd_start)

    p_stop = sub.add_parser("stop", help="signal and stop the daemon")
    p_stop.set_defaults(func=_cmd_stop)

    p_status = sub.add_parser("status", help="report daemon health + metrics")
    p_status.set_defaults(func=_cmd_status)

    p_restart = sub.add_parser(
        "restart", help="stop then start --background",
    )
    p_restart.add_argument(
        "--background", action="store_true", default=True,
        help="(noop — restart is always backgrounded)",
    )
    p_restart.set_defaults(func=_cmd_restart)
