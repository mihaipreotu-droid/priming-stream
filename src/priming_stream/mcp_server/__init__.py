"""mcp_server — local stdio MCP server over the read-only graph operations."""
from priming_stream.mcp_server.server import main
from priming_stream.mcp_server.tools import TOOL_SCHEMAS, TOOLS

__all__ = ["TOOLS", "TOOL_SCHEMAS", "main"]
