"""Local stdio MCP server exposing the v0.7-x-vec-index records substrate.

Registers the tools from :mod:`priming_stream.mcp_server.tools` and dispatches
calls to their handlers. Every tool opens SQLite read-only via the
``mode=ro`` URI — v0.7-x ships no write tools (durable writes happen
through the ``/prime-sleep`` skill in an active Claude Code session, not
through MCP). The vec_index (fastembed + ChromaDB) is a process-global
singleton lazily constructed inside ``tools.py``.

Importing this module never starts the server; ``main()`` runs the stdio
loop.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from priming_stream.core.config import load_config
from priming_stream.core.graph_repo import GraphRepo
from priming_stream.core.paths import resolve_paths

from priming_stream.mcp_server.tools import TOOLS, TOOL_SCHEMAS, WRITE_TOOLS
from priming_stream.mcp_server.usage_log import log_usage

SERVER_NAME = "priming-stream-graph"


def _connect_readonly(graph_db: Path) -> sqlite3.Connection:
    """Open the active graph strictly read-only.

    The ``mode=ro`` URI makes SQLite reject every write at the engine level —
    a hard guarantee on top of the handlers already being read-only.
    """
    uri = f"file:{graph_db.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def dispatch_tool(name: str, arguments: dict, graph_db: Path):
    """Run one tool against a fresh read-only connection.

    v0.7-x has no write tools (``WRITE_TOOLS`` is empty); the branch is
    kept for forward-compatibility if a record-write tool is ever added.
    """
    handler = TOOLS.get(name)
    if handler is None:
        raise ValueError(f"unknown tool: {name!r}")

    if name in WRITE_TOOLS:
        # Currently unreachable; left here so a future write tool plugs in
        # without a server.py edit.
        from priming_stream.core.db import connect as connect_rw
        conn = connect_rw(graph_db)
    else:
        conn = _connect_readonly(graph_db)
    try:
        repo = GraphRepo(conn)
        started = time.perf_counter()
        result = handler(repo, arguments or {})
        # Active-use telemetry (Phase-5 calibration input): record this read
        # AFTER a successful handler call. log_usage swallows its own errors;
        # the extra guard here is belt-and-suspenders so a future bug that
        # escapes it can still never break or slow the tool call.
        try:
            log_usage(
                name, arguments or {}, result,
                (time.perf_counter() - started) * 1000.0, graph_db,
            )
        except Exception:
            pass
        return result
    finally:
        conn.close()


def build_server(graph_db: Path) -> Server:
    """Build (but do not run) the MCP server bound to ``graph_db``."""
    server: Server = Server(SERVER_NAME)

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name=name,
                description=TOOL_SCHEMAS[name]["description"],
                inputSchema=TOOL_SCHEMAS[name]["inputSchema"],
            )
            for name in TOOLS
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        result = dispatch_tool(name, arguments, graph_db)
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    return server


async def _run(graph_db: Path) -> None:
    server = build_server(graph_db)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main() -> None:
    """Entry point: resolve config and graph path, then run the stdio server."""
    import asyncio

    cfg = load_config()
    paths = resolve_paths(cfg)
    asyncio.run(_run(paths.graph_db))


if __name__ == "__main__":
    main()
