"""Ingestion adapters — read raw transcripts and yield normalized chunks.

v0.7-x: the only adapter left is ``ClaudeCodeAdapter`` (line-JSON transcripts
under ``~/.claude/projects``). The legacy ``ClaudeDesktopAdapter`` (a stub
redirect) and ``run_ingestion`` (write chunks into the episodic store) are
gone — coldstart now materializes chunks directly via ``ingest.materialize``.
"""
from __future__ import annotations

from .base import Adapter
from .claude_ai_export import ClaudeAiExportAdapter
from .claude_code import ClaudeCodeAdapter

__all__ = [
    "Adapter",
    "ClaudeCodeAdapter",
    "ClaudeAiExportAdapter",
]
