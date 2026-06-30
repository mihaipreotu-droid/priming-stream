"""Filesystem layout for Priming Stream runtime state, derived from config.

v0.7-b drops the ``corpus/convos/`` + ``corpus/sources/`` watch folders
(retracted concept — no manual copy-paste flow expected) and replaces them
with a single ``storage/exports/`` watch folder for claude.ai data-export
archives. The sleep cycle scans it each run; recognized archives are
ingested then moved to ``storage/exports/processed/``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from priming_stream.core.config import Config


@dataclass
class ProjectPaths:
    storage_dir: Path
    graph_db: Path
    nodes_dir: Path
    episodic_dir: Path
    vocab_log: Path
    exports_dir: Path
    exports_processed_dir: Path
    vec_index_dir: Path
    # v0.7-x-vec-index: storage/corpus/ holds the episodic .md layer
    # (materialized chunk exports under imports/) + per-cycle scratch.
    # Renamed from the historical ``qmd-corpus`` after qmd was removed as
    # a runtime dependency.
    corpus_dir: Path
    # SQL-canonical (2026-06-12): records live in SQLite; this path is the
    # RETIRED per-record .md location — kept only so legacy/archival tooling
    # can name it. Never created or written by current code.
    corpus_records_dir: Path
    corpus_imports_dir: Path
    corpus_cursor_path: Path
    corpus_sleep_manifest_path: Path


def resolve_paths(cfg: Config, project_root: Path | None = None) -> ProjectPaths:
    # Resolution order for storage_dir:
    #   1. ``PRIMING_STREAM_STORAGE_DIR`` env var (used by tests for isolation, by ad-hoc
    #      cli overrides). Always treated as absolute (resolved if relative).
    #   2. ``cfg.paths.storage_dir`` if absolute — use as-is.
    #   3. Explicit ``project_root`` arg + relative config → resolve under it.
    #      CLI entry points (``coldstart``, ``sleep-prepare`` etc.) pass
    #      ``Path.cwd()`` here so the user can run the CLI from any project
    #      and operate on that project's storage.
    #   4. No explicit project_root + relative config → anchor at the
    #      ``settings.toml`` parent (the Priming Stream project root, regardless of
    #      cwd). This is the path the bridge hook takes when invoked from
    #      a foreign cwd (e.g., another Claude Code project) — it should
    #      still find the Priming Stream substrate.
    import os
    override = os.environ.get("PRIMING_STREAM_STORAGE_DIR")
    if override:
        storage_dir = Path(override).resolve()
    else:
        raw = Path(cfg.paths.storage_dir)
        if raw.is_absolute():
            storage_dir = raw
        elif project_root is not None:
            storage_dir = Path(project_root) / raw
        else:
            # Anchor at the directory containing settings.toml — found by
            # walking up from this module (which is always inside the Priming Stream
            # source tree). Falls back to cwd if settings.toml isn't found.
            from priming_stream.core.config import _find_settings
            settings_path = _find_settings()
            if settings_path is not None:
                anchor = settings_path.parent.parent
            else:
                anchor = Path.cwd()
            storage_dir = anchor / raw

    exports_raw = Path(cfg.paths.exports_dir)
    if exports_raw.is_absolute():
        exports_dir = exports_raw
    else:
        exports_dir = storage_dir / exports_raw

    # v0.7-x-vec-index: ChromaDB persistent root. Same convention as
    # ``exports_dir`` — absolute paths used as-is; relative paths resolve
    # under ``storage_dir``.
    vec_raw = Path(cfg.vec_index.persist_dir)
    if vec_raw.is_absolute():
        vec_index_dir = vec_raw
    else:
        vec_index_dir = storage_dir / vec_raw

    corpus_dir = storage_dir / "corpus"
    return ProjectPaths(
        storage_dir=storage_dir,
        graph_db=storage_dir / "graph.db",
        nodes_dir=storage_dir / "nodes",
        episodic_dir=storage_dir / "episodic",
        vocab_log=storage_dir / "vocabulary_review.log",
        exports_dir=exports_dir,
        exports_processed_dir=exports_dir / "processed",
        vec_index_dir=vec_index_dir,
        corpus_dir=corpus_dir,
        corpus_records_dir=corpus_dir / "records",
        corpus_imports_dir=corpus_dir / "imports",
        corpus_cursor_path=corpus_dir / "_cursor.json",
        corpus_sleep_manifest_path=corpus_dir / "_sleep_manifest.json",
    )


def ensure_dirs(paths: ProjectPaths) -> None:
    paths.storage_dir.mkdir(parents=True, exist_ok=True)
    paths.nodes_dir.mkdir(parents=True, exist_ok=True)
    paths.episodic_dir.mkdir(parents=True, exist_ok=True)
    paths.exports_dir.mkdir(parents=True, exist_ok=True)
    paths.exports_processed_dir.mkdir(parents=True, exist_ok=True)
    paths.vec_index_dir.mkdir(parents=True, exist_ok=True)
    paths.corpus_dir.mkdir(parents=True, exist_ok=True)
    # corpus_records_dir deliberately NOT created — the per-record .md layer
    # is retired (SQL-canonical); records live in SQLite.
    paths.corpus_imports_dir.mkdir(parents=True, exist_ok=True)


def migrate_qmd_corpus_to_corpus(storage_dir: Path) -> bool:
    """One-time rename: ``storage/qmd-corpus/`` -> ``storage/corpus/``.

    Idempotent. Returns True iff a rename happened. Safe to call from
    ``prime init`` and ``vec-index-rebuild``: if ``storage/corpus``
    already exists or no legacy dir is present, this is a no-op.

    If BOTH legacy and target exist (half-done previous run, or manual
    mkdir), prints a stderr warning and returns False — auto-merging the
    two trees is too risky to do silently.
    """
    import sys

    legacy = storage_dir / "qmd-corpus"
    target = storage_dir / "corpus"
    if not legacy.exists():
        return False
    if target.exists():
        print(
            "warning: both storage/qmd-corpus/ and storage/corpus/ exist. "
            "Migration skipped.\n"
            "         Inspect both manually; rename or remove the legacy "
            "directory before\n"
            "         the next run.",
            file=sys.stderr,
        )
        return False
    legacy.rename(target)
    return True
