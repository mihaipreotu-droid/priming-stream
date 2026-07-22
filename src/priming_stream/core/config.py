"""Configuration loading: settings.toml overlaid onto frozen defaults (v0.7-x).

The v0.7-x-vec-index substrate uses ``[paths]``, ``[sleep]``, ``[bridge]``,
``[llm]``, ``[mcp]``, and ``[vec_index]``. The old PPR / genesis / injection
/ aliases / corpus / weights sections are gone with the rest of the
nodes-and-edges substrate; ``[qmd]`` is gone with the retrieval-transport
swap (qmd CLI -> fastembed + ChromaDB in-process).
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PathsConfig:
    storage_dir: str = "storage"
    # Watch folder for claude.ai data-export archives. Relative paths are
    # resolved under storage_dir; absolute paths are used as-is.
    exports_dir: str = "exports"


@dataclass
class SleepConfig:
    idle_minutes: int = 30
    chunk_max_turns: int = 120
    mutex_timeout_s: int = 300
    # Per-chunk char budget enforced at ingestion time (kept for the
    # adapter's ``split_bursts`` helper that materializes coldstart chunks).
    chunk_max_chars: int = 30_000


@dataclass
class BridgeConfig:
    # v0.7-x W-D: adaptive spreading activation knobs (spec §3 / brief §D).
    # Multi-hop vec_index search with multiplicative decay. Defaults are the
    # checked-in starting point; Phase 5 calibrates against the real corpus.
    decay: float = 0.8
    min_score: float = 0.3
    frontier_cap: int = 10
    k_per_query: int = 10
    max_hops: int = 4
    # Raw-spread output cap for the deliberate pull surfaces (graph_spread_op
    # and the MCP graph_spread tool — the legacy single-seed spread() is gone);
    # equals the semantic budget (bucket A).
    max_records: int = 20
    # Read-time bridge restructuring (two-seed walk + two-bucket; see ARCHITECTURE.md).
    # Frozen defaults — Phase 5 calibrates. recency_* drive A.5b/A.5c (selection-
    # side, bucket A only); bucket_* drive the A.2 two-bucket split.
    recency_strength: float = 0.25      # A.5b, [0,1] off->gentle->hard. ON by default.
    recency_age_span_days: int = 180    # A.5b age normalization span (days).
    recency_p_max: float = 0.5          # A.5b penalty ceiling.
    bucket_total: int = 25              # shared priming budget (semantic + lexical).
    bucket_lexical: int = 5             # bucket B cap; semantic = bucket_total - bucket_lexical.
    recency_filter_cutoff: str = ""     # A.5c hard cutoff date; "" = off (stretch).
    # Item 3.3 cross-turn dedup: a record primed in the last N turns of the same
    # session is not re-emitted; the freed slot backfills from the tail (queue
    # advances). Window is on TURNS (not permanent), so a still-relevant record
    # re-emits after N turns. 0 = off. Env PRIMING_STREAM_DEDUP_OFF also disables.
    # Applied in the hook (which reads the per-session echo history); the daemon
    # just filters the ids it receives. Validated on replay: at N=10, in-window
    # repeats 31%->0, 0 records lost, +distal surfaced.
    dedup_window_turns: int = 10
    # Seed char budget (2026-07-21, paired with the 2s client deadline):
    # user-first input allocation for the semantic seed. The user prompt is
    # NEVER truncated; ``prev_assistant_text`` (P5) takes what remains up to
    # this total. Sized so embed stays under the 2s client deadline under
    # int8 (~0.28ms/char + walk 50-160ms + ~313ms fixed hook overhead).
    # NOT an adaptive deadline — deadline stays fixed; input is allocated
    # to fit under it.
    seed_char_budget: int = 5000
    # Turn-gate (2026-07-21): ONE mechanism, TWO outputs — FULL or WHISPER,
    # zero silence. Applies ONLY to
    # the hook push path (requests carrying turn features) — MCP/CLI
    # deliberate pulls stay full.
    # Whisper = top ``whisper_k`` semantic + top ``whisper_lex_k`` lexical,
    # uniform across its three triggers: turn top rank-score under
    # ``turn_floor`` (whisper-floor, rendered with a weak-field marker; the
    # 0.40 default sits above the measured top score of a deliberately
    # unrelated control prompt, 0.285), execution regime
    # (``tool_density >= regime_density``), and <task-notification> turns.
    # whisper_k=5 not 3: known-valuable records cluster at rank 4-5 (5/12
    # measured; two-lineage rank redistribution). whisper_lex_k=3 not 0:
    # dense/notification turns are exactly where exact-identifier FTS wins
    # (2/14 valuable appearances came via the lexical bucket).
    # kickoff_turns = first N turns of a session are exempt (full,
    # unconditional). turn_floor = 0 -> ENTIRE gate off (one-knob rollback).
    turn_floor: float = 0.0
    regime_density: float = 0.6
    whisper_k: int = 5
    whisper_lex_k: int = 3
    kickoff_turns: int = 3


@dataclass
class LlmConfig:
    # Empty string = use SDK/Claude Code default model.
    # Set to e.g. "sonnet" or "opus" to override per-run.
    model: str = ""
    # Env var the SDK reads for OAuth (Claude Pro/Max subscription auth).
    # Set externally; this name is informational. Generated by
    # ``claude setup-token``.
    auth_token_env: str = "CLAUDE_CODE_OAUTH_TOKEN"


@dataclass
class McpConfig:
    read_only: bool = True


@dataclass
class VecIndexConfig:
    """v0.7-x-vec-index — fastembed + ChromaDB transport.

    ``persist_dir`` is the ChromaDB persistent root. Relative paths resolve
    under ``paths.storage_dir`` (same convention as ``exports_dir``);
    absolute paths are used as-is. ``model_name`` is passed through to
    ``fastembed.TextEmbedding``; the default MATCHES the live collection
    (int8 bge-m3, 1024-dim — custom-registered in integrations/vec_index.py)
    so a missing/stripped settings.toml cannot silently query the 1024-dim
    collection with a different-dimension model (2026-07-21 review;; the old
    MiniLM-384 default was exactly that footgun).
    """
    persist_dir: str = "vec_index/chroma"
    model_name: str = "onnx-community/bge-m3-ONNX-int8"


@dataclass
class Config:
    paths: PathsConfig
    sleep: SleepConfig
    bridge: BridgeConfig
    llm: LlmConfig
    mcp: McpConfig
    vec_index: VecIndexConfig


def _defaults() -> Config:
    return Config(
        paths=PathsConfig(),
        sleep=SleepConfig(),
        bridge=BridgeConfig(),
        llm=LlmConfig(),
        mcp=McpConfig(),
        vec_index=VecIndexConfig(),
    )


def _find_settings() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "config" / "settings.toml"
        if candidate.is_file():
            return candidate
    return None


def _overlay(section, data: dict) -> None:
    for key in vars(section):
        if key in data:
            setattr(section, key, data[key])


def load_config(path: Path | None = None) -> Config:
    cfg = _defaults()

    if path is None:
        path = _find_settings()
        if path is None:
            return cfg

    path = Path(path)
    if not path.is_file():
        return cfg

    with path.open("rb") as fh:
        raw = tomllib.load(fh)

    _overlay(cfg.paths, raw.get("paths", {}))
    _overlay(cfg.sleep, raw.get("sleep", {}))
    _overlay(cfg.bridge, raw.get("bridge", {}))
    _overlay(cfg.llm, raw.get("llm", {}))
    _overlay(cfg.mcp, raw.get("mcp", {}))
    _overlay(cfg.vec_index, raw.get("vec_index", {}))

    return cfg
