"""Read-only MCP tool handlers over the v0.7-x-vec-index records substrate.

Each handler is ``handler(repo, args) -> JSON-compatible``. Handlers all
share a process-global :class:`RecordsVecIndex` instance built once on
first call (the MCP server is long-lived; we pay the embedder init cost
~once at first tool dispatch instead of per call).

v0.7-x-vec-index surface (spec §4.6 / brief §G):

    graph_search_records       — vec_index search on records collection
    graph_records              — fetch one record by id
    graph_chunk_around_anchor  — §16.6 verification path (file IO only)
    graph_spread               — bridge.spread on arbitrary text
    graph_stats                — records + sleep + vec_index size
    graph_disambiguate         — bridge over LLM reformulation
    graph_salient_context      — pull-bridge for Claude Desktop

Dropped vs v0.7: ``graph_search_node``, ``graph_neighbors``, ``graph_path``,
``graph_node_detail``, ``index_document``.
Dropped vs v0.7-x: ``graph_search_chunks`` — chunk evidence is reachable
via ``graph_chunk_around_anchor`` (tier-2 from a record).
"""
from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any, Callable

from priming_stream.bridge.working_set import build_priming, priming_items
from priming_stream.core.config import load_config
from priming_stream.core.graph_repo import GraphRepo
from priming_stream.core.paths import resolve_paths
from priming_stream.daemon.render import render_buckets
from priming_stream.graph_ops import (
    graph_chunk_around_anchor as _op_chunk_around_anchor,
    graph_records as _op_records,
    graph_search_lexical as _op_search_lexical,
    graph_search_records as _op_search_records,
    graph_stats as _op_stats,
)
from priming_stream.integrations.vec_index import RecordsVecIndex


# Per spec: MCP server is read-only in v0.7-x. No write tools registered.
WRITE_TOOLS: set[str] = set()


# -- shared boilerplate -------------------------------------------------------


@cache
def _get_vec_index() -> RecordsVecIndex:
    """Process-global :class:`RecordsVecIndex` singleton.

    ``functools.cache`` ensures only one instance is constructed per
    process. The fastembed model is lazy (loaded on first ``search`` /
    ``add_record`` call), so the cache lookup is cheap until then.
    """
    cfg = load_config()
    paths = resolve_paths(cfg)
    return RecordsVecIndex(paths.vec_index_dir, cfg.vec_index.model_name)


def _corpus_paths() -> tuple[Path, Path]:
    """Return ``(storage_dir, corpus_dir)`` for the chunk-anchor handler."""
    cfg = load_config()
    paths = resolve_paths(cfg)
    return paths.storage_dir, paths.corpus_dir


def _get_conn(repo: GraphRepo):
    """Extract the underlying SQLite connection from ``repo``.

    ``GraphRepo`` exposes its connection as the public ``conn`` attribute.
    ``lexical_bucket`` needs the raw connection for its FTS5 SELECT; the
    MCP dispatch opens the connection and wraps it in a ``GraphRepo``, so
    we pull it back out here rather than opening a second connection.
    """
    return repo.conn


# -- handlers -----------------------------------------------------------------


def graph_search_records(repo: GraphRepo, args: dict) -> list[dict]:
    query_text = str(args.get("query_text", "") or "")
    k = int(args.get("k", 10))
    return _op_search_records(query_text, k, _get_vec_index(), repo)


def graph_search_lexical(repo: GraphRepo, args: dict) -> list[dict]:
    query_text = str(args.get("query_text", "") or "")
    k = int(args.get("k", 10))
    mode = str(args.get("mode", "and") or "and")
    return _op_search_lexical(query_text, k, mode, repo)


def graph_records(repo: GraphRepo, args: dict) -> dict | None:
    record_id = str(args.get("record_id", "") or "")
    return _op_records(record_id, repo)


def graph_chunk_around_anchor(repo: GraphRepo, args: dict) -> dict:
    record_id = str(args.get("record_id", "") or "")
    window = int(args.get("window", 200))
    storage_dir, corpus_dir = _corpus_paths()
    return _op_chunk_around_anchor(
        record_id, window, repo, storage_dir, corpus_dir,
    )


def graph_spread(repo: GraphRepo, args: dict) -> list[dict]:
    """Run the bridge A-pipeline on arbitrary text and return ranked records.

    Uses ``walk_two_seeds(text, "")`` (single-seed: prompt only, no prev)
    capped to ``cfg.bridge.max_records`` — the raw-spread output cap for
    deliberate MCP/CLI surfaces. Output carries ``source_date`` and ``kind``
    for parity with the search tools.
    """
    text = str(args.get("text", "") or "")
    if not text.strip():
        return []
    cfg = load_config()
    result = build_priming(
        text, "",
        vec_index=_get_vec_index(),
        repo=repo,
        conn=_get_conn(repo),
        cfg=cfg.bridge,
    )
    sem, lex = priming_items(result)
    # Merge semantic + lexical into a single ranked list capped at max_records.
    # Semantic items come first (higher-quality activation); lexical items
    # fill the remaining cap. Rank is re-numbered after merge.
    combined = (sem + lex)[: cfg.bridge.max_records]
    for i, item in enumerate(combined):
        item["rank"] = i + 1
    return combined


def graph_stats(repo: GraphRepo, args: dict) -> dict:
    _ = args
    return _op_stats(repo, _get_vec_index())


def graph_disambiguate(repo: GraphRepo, args: dict) -> str:
    """LLM-triggered disambiguation pull-bridge.

    The LLM calls this when the user's prompt uses ambiguous references
    (pronouns, deictics, vague objects) and passes a canonical
    reformulation in ``text``. Runs the full A-pipeline (two-bucket:
    semantic + lexical) and returns the rendered two-bucket markdown block.
    """
    if not isinstance(args, dict):
        return "(invalid args)"
    text = args.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return "(empty text)"

    cfg = load_config()
    result = build_priming(
        text, "",
        vec_index=_get_vec_index(),
        repo=repo,
        conn=_get_conn(repo),
        cfg=cfg.bridge,
    )
    if not result.semantic and not result.lexical:
        return "(no salient context for this text)"
    sem, lex = priming_items(result)
    return render_buckets(sem, lex)


def graph_salient_context(repo: GraphRepo, args: dict) -> str:
    """Stateless pull-bridge for Claude Desktop and clients without hooks.

    Runs the full A-pipeline (two-bucket: semantic + lexical) on ``message``
    and returns the rendered two-bucket markdown block — same shape the
    ``UserPromptSubmit`` hook injects.
    """
    if not isinstance(args, dict):
        return "(invalid args)"
    message = args.get("message", "")
    if not isinstance(message, str) or not message:
        return "(empty message)"

    cfg = load_config()
    result = build_priming(
        message, "",
        vec_index=_get_vec_index(),
        repo=repo,
        conn=_get_conn(repo),
        cfg=cfg.bridge,
    )
    if not result.semantic and not result.lexical:
        return "(no salient context for this message)"
    sem, lex = priming_items(result)
    return render_buckets(sem, lex)


# -- registry + schemas -------------------------------------------------------


TOOLS: dict[str, Callable[[GraphRepo, dict], Any]] = {
    "graph_search_records": graph_search_records,
    "graph_search_lexical": graph_search_lexical,
    "graph_records": graph_records,
    "graph_chunk_around_anchor": graph_chunk_around_anchor,
    "graph_spread": graph_spread,
    "graph_stats": graph_stats,
    "graph_disambiguate": graph_disambiguate,
    "graph_salient_context": graph_salient_context,
}


TOOL_SCHEMAS: dict[str, dict] = {
    "graph_search_records": {
        "description": (
            "SEMANTIC (vector similarity) search over the records substrate — "
            "the associative / conceptual recall channel. Returns [{record_id, "
            "summary, score, source_uri, source_date, kind}] ordered by score "
            "descending. Reach for it PROACTIVELY to recall past conversation "
            "moments by MEANING when the already-injected priming context did "
            "not surface what the user is asking for (e.g. 'do you remember "
            "when we discussed X', paraphrased / conceptual recall). "
            "Register-tolerant, but it misses or buries bare exact terms, "
            "names, and citations — for those use graph_search_lexical. When "
            "unsure which channel fits, call BOTH (complementary, cheap, "
            "read-only) and merge. Records are LLM-distilled summaries; fetch "
            "the source chunk via graph_chunk_around_anchor to verify."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query_text": {"type": "string"},
                "k": {"type": "integer", "default": 10},
            },
            "required": ["query_text"],
        },
    },
    "graph_search_lexical": {
        "description": (
            "LEXICAL (FTS5 BM25, keyword/term) search over the records "
            "substrate — the exact-term recall channel, counterpart of "
            "graph_search_records (semantic). Reach for it PROACTIVELY to find "
            "a record by an EXACT term, proper name, or citation the user "
            "remembers (e.g. 'what was that paper called', 'what did Collins "
            "say about Y') — cases dense-embedding similarity misses or buries, "
            "and which the injected priming did not already satisfy. YOU "
            "compose the query: extract the search INTENT — the specific "
            "term(s), not the user's whole sentence — and pick a mode. Returns "
            "[{record_id, summary, score, source_uri, source_date, kind}] "
            "best-first (score = raw bm25, more negative = better; trust the "
            "order). mode: 'and' (default — every term must appear, precision), "
            "'or' (any term, recall), 'phrase' (exact ordered phrase). When "
            "unsure whether the user wants an exact term or a concept, also "
            "call graph_search_records and merge."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query_text": {"type": "string"},
                "mode": {
                    "type": "string",
                    "enum": ["and", "or", "phrase"],
                    "default": "and",
                },
                "k": {"type": "integer", "default": 10},
            },
            "required": ["query_text"],
        },
    },
    "graph_records": {
        "description": (
            "Fetch one record by id. Returns {id, source_uri, "
            "anchor_offset_start, anchor_offset_end, summary, created_at} "
            "or null if the id is unknown."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "record_id": {"type": "string"},
            },
            "required": ["record_id"],
        },
    },
    "graph_chunk_around_anchor": {
        "description": (
            "Tier-2 verification: read the source file the record anchors "
            "to and return text around the anchor (±window characters). "
            "If the record has no anchor offsets (chunk-level record), "
            "returns the whole file. Records prime; chunks verify."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "record_id": {"type": "string"},
                "window": {"type": "integer", "default": 200},
            },
            "required": ["record_id"],
        },
    },
    "graph_spread": {
        "description": (
            "Run the full bridge A-pipeline (two-seed walk + two-bucket) on "
            "arbitrary text. Returns [{record_id, summary, rank, source_date, "
            "kind, source_uri, anchor_start, anchor_end}] capped at "
            "cfg.bridge.max_records (raw-spread output cap for deliberate "
            "MCP/CLI surfaces). Semantic records first, lexical appended; "
            "rank is 1-based over the merged list. Use graph_salient_context "
            "for the rendered markdown block instead."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
            },
            "required": ["text"],
        },
    },
    "graph_stats": {
        "description": (
            "Substrate health: {records_count, last_sleep_cycle, "
            "vec_index_size}. records_count is the SQLite count; "
            "last_sleep_cycle is the most recent sleep run (or null); "
            "vec_index_size is the ChromaDB records-collection count."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    "graph_disambiguate": {
        "description": (
            "Disambiguation pull-bridge. Call when the user's prompt uses "
            "ambiguous references (pronouns, deictics, vague objects); in "
            "`text` pass your best canonical reformulation. Returns the "
            "rendered salient-context markdown block for that reformulation. "
            "Do NOT call on acknowledgements ('ok', 'mersi') or prompts "
            "where the object is named explicitly."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": (
                        "Your canonical reformulation of the user's "
                        "ambiguous reference."
                    ),
                },
            },
            "required": ["text"],
        },
    },
    "graph_salient_context": {
        "description": (
            "Pull-bridge for Claude Desktop and other clients without "
            "hooks. Given the user's message, runs the bridge spreading "
            "walk and returns the resulting salient-context markdown "
            "block. Stateless (no session continuity)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The user's most recent message.",
                },
            },
            "required": ["message"],
        },
    },
}
