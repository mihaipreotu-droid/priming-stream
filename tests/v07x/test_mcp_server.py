"""v0.7-x W-F: MCP server smoke + tool registry.

Verifies:
- ``build_server`` imports and instantiates without error (F1).
- The registered tool set is exactly the v0.7-x surface (F2).
- Dropped tools from v0.7 are NOT registered.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from priming_stream.mcp_server import server as server_mod
from priming_stream.mcp_server.tools import TOOL_SCHEMAS, TOOLS


_EXPECTED = {
    "graph_search_records",
    "graph_search_lexical",
    "graph_records",
    "graph_chunk_around_anchor",
    "graph_spread",
    "graph_stats",
    "graph_disambiguate",
    "graph_salient_context",
}

_DROPPED = {
    "graph_search_node",
    "graph_neighbors",
    "graph_path",
    "graph_node_detail",
    "index_document",
    # v0.7-x-vec-index: chunks search dropped — reach chunks via
    # graph_chunk_around_anchor (tier-2).
    "graph_search_chunks",
}


def test_build_server_succeeds(tmp_path):
    """F1: build_server runs without exception on an arbitrary path.

    The path does not need to exist — the server only connects when a
    tool is dispatched.
    """
    server = server_mod.build_server(tmp_path / "graph.db")
    assert server is not None
    assert server.name == server_mod.SERVER_NAME


def test_tool_registry_contains_v07x_surface():
    """F2: every v0.7-x tool is registered."""
    assert _EXPECTED.issubset(set(TOOLS.keys()))


def test_tool_registry_drops_v07_tools():
    """F2: dropped tools are not registered."""
    assert _DROPPED.isdisjoint(set(TOOLS.keys()))


def test_tool_registry_is_exactly_v07x_surface():
    """The registry contains the v0.7-x set and nothing else."""
    assert set(TOOLS.keys()) == _EXPECTED


def test_each_tool_has_a_schema():
    assert set(TOOLS.keys()) == set(TOOL_SCHEMAS.keys())


def test_each_schema_has_required_fields():
    for name, schema in TOOL_SCHEMAS.items():
        assert "description" in schema, name
        assert "inputSchema" in schema, name
        assert schema["inputSchema"]["type"] == "object", name


def test_dispatch_unknown_tool_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown tool"):
        server_mod.dispatch_tool("not_a_tool", {}, tmp_path / "graph.db")
