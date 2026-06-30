"""Unit — paths: relative/absolute resolution, directory creation (v0.7-b)."""
from pathlib import Path

from priming_stream.core.config import load_config
from priming_stream.core.paths import ensure_dirs, resolve_paths


def test_relative_storage_resolved_against_root(tmp_path):
    cfg = load_config(tmp_path / "absent.toml")  # defaults: storage_dir="storage"
    paths = resolve_paths(cfg, project_root=tmp_path)
    assert paths.storage_dir == tmp_path / "storage"
    assert paths.graph_db == tmp_path / "storage" / "graph.db"
    assert paths.nodes_dir == tmp_path / "storage" / "nodes"
    assert paths.episodic_dir == tmp_path / "storage" / "episodic"
    assert paths.vocab_log == tmp_path / "storage" / "vocabulary_review.log"


def test_absolute_storage_used_as_is(tmp_path):
    cfg = load_config(tmp_path / "absent.toml")
    abs_dir = (tmp_path / "abs storage").resolve()
    cfg.paths.storage_dir = str(abs_dir)
    paths = resolve_paths(cfg, project_root=tmp_path)
    assert paths.storage_dir == Path(abs_dir)


def test_ensure_dirs_creates_layout(tmp_path):
    cfg = load_config(tmp_path / "absent.toml")
    paths = resolve_paths(cfg, project_root=tmp_path)
    ensure_dirs(paths)
    assert paths.storage_dir.is_dir()
    assert paths.nodes_dir.is_dir()
    assert paths.episodic_dir.is_dir()


# -- v0.7-b exports watch folder -----------------------------------------

def test_exports_paths_derived(tmp_path):
    """Default exports_dir resolves under storage_dir."""
    cfg = load_config(tmp_path / "absent.toml")
    paths = resolve_paths(cfg, project_root=tmp_path)
    assert paths.exports_dir == tmp_path / "storage" / "exports"
    assert paths.exports_processed_dir == \
        tmp_path / "storage" / "exports" / "processed"


def test_absolute_exports_dir_used_as_is(tmp_path):
    cfg = load_config(tmp_path / "absent.toml")
    abs_exports = (tmp_path / "elsewhere" / "exports").resolve()
    cfg.paths.exports_dir = str(abs_exports)
    paths = resolve_paths(cfg, project_root=tmp_path)
    assert paths.exports_dir == abs_exports
    assert paths.exports_processed_dir == abs_exports / "processed"


def test_ensure_dirs_creates_exports_layout(tmp_path):
    cfg = load_config(tmp_path / "absent.toml")
    paths = resolve_paths(cfg, project_root=tmp_path)
    ensure_dirs(paths)
    assert paths.exports_dir.is_dir()
    assert paths.exports_processed_dir.is_dir()


# Verify retracted concept gone — convos_dir / sources_dir no longer attrs.

def test_legacy_corpus_paths_absent(tmp_path):
    cfg = load_config(tmp_path / "absent.toml")
    paths = resolve_paths(cfg, project_root=tmp_path)
    assert not hasattr(paths, "convos_dir")
    assert not hasattr(paths, "sources_dir")
    # ``corpus_dir`` came back in v0.7-x-vec-index as the renamed
    # ``storage/qmd-corpus/``. It's expected to exist as an attribute
    # now — see test_corpus_paths_derived below.


# -- v0.7-x-vec-index corpus dir tree ------------------------------------


def test_corpus_paths_derived(tmp_path):
    cfg = load_config(tmp_path / "absent.toml")
    paths = resolve_paths(cfg, project_root=tmp_path)
    assert paths.corpus_dir == tmp_path / "storage" / "corpus"
    assert paths.corpus_records_dir == tmp_path / "storage" / "corpus" / "records"
    assert paths.corpus_imports_dir == tmp_path / "storage" / "corpus" / "imports"
    assert paths.corpus_cursor_path == \
        tmp_path / "storage" / "corpus" / "_cursor.json"


def test_ensure_dirs_creates_corpus_layout(tmp_path):
    from priming_stream.core.paths import ensure_dirs
    cfg = load_config(tmp_path / "absent.toml")
    paths = resolve_paths(cfg, project_root=tmp_path)
    ensure_dirs(paths)
    assert paths.corpus_dir.is_dir()
    # SQL-canonical: the per-record .md dir is retired — never created.
    assert not paths.corpus_records_dir.exists()
    assert paths.corpus_imports_dir.is_dir()


def test_migrate_qmd_corpus_to_corpus(tmp_path):
    """One-time rename: legacy ``qmd-corpus/`` -> ``corpus/`` if present."""
    from priming_stream.core.paths import migrate_qmd_corpus_to_corpus

    storage = tmp_path / "storage"
    storage.mkdir()
    legacy = storage / "qmd-corpus"
    legacy.mkdir()
    (legacy / "marker.txt").write_text("legacy", encoding="utf-8")

    moved = migrate_qmd_corpus_to_corpus(storage)
    assert moved is True
    assert not legacy.exists()
    assert (storage / "corpus" / "marker.txt").read_text(encoding="utf-8") == "legacy"


def test_migrate_qmd_corpus_idempotent(tmp_path):
    """Second call is a no-op (returns False)."""
    from priming_stream.core.paths import migrate_qmd_corpus_to_corpus

    storage = tmp_path / "storage"
    storage.mkdir()
    assert migrate_qmd_corpus_to_corpus(storage) is False  # neither dir
    (storage / "corpus").mkdir()
    assert migrate_qmd_corpus_to_corpus(storage) is False  # only target


def test_migrate_qmd_corpus_skips_when_target_already_exists(tmp_path):
    """If both legacy and target exist, don't clobber the target."""
    from priming_stream.core.paths import migrate_qmd_corpus_to_corpus

    storage = tmp_path / "storage"
    storage.mkdir()
    (storage / "qmd-corpus").mkdir()
    (storage / "corpus").mkdir()
    (storage / "corpus" / "keep.txt").write_text("keep", encoding="utf-8")
    assert migrate_qmd_corpus_to_corpus(storage) is False
    # Both directories still exist; corpus is unchanged.
    assert (storage / "qmd-corpus").exists()
    assert (storage / "corpus" / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_migrate_qmd_corpus_warns_on_collision(tmp_path, capsys):
    """Collision (both legacy and target exist) prints a stderr warning."""
    from priming_stream.core.paths import migrate_qmd_corpus_to_corpus

    storage = tmp_path / "storage"
    storage.mkdir()
    (storage / "qmd-corpus").mkdir()
    (storage / "qmd-corpus" / "legacy.txt").write_text("legacy", encoding="utf-8")
    (storage / "corpus").mkdir()
    (storage / "corpus" / "keep.txt").write_text("keep", encoding="utf-8")

    moved = migrate_qmd_corpus_to_corpus(storage)
    assert moved is False
    captured = capsys.readouterr()
    assert "qmd-corpus" in captured.err
    assert "corpus" in captured.err
    assert "skipped" in captured.err.lower() or "manual" in captured.err.lower()


# -- v0.7-x-vec-index: vec_index_dir resolution --------------------------


def test_vec_index_dir_resolves_under_storage_dir(tmp_path):
    """Relative ``vec_index.persist_dir`` resolves under storage_dir,
    matching the ``exports_dir`` convention. Honors an absolute
    storage_dir.
    """
    cfg = load_config(tmp_path / "absent.toml")
    abs_storage = (tmp_path / "abs storage").resolve()
    cfg.paths.storage_dir = str(abs_storage)
    cfg.vec_index.persist_dir = "vec/chroma"

    paths = resolve_paths(cfg, project_root=tmp_path)
    assert paths.vec_index_dir == abs_storage / "vec" / "chroma"


def test_absolute_vec_index_dir_used_as_is(tmp_path):
    """Absolute ``vec_index.persist_dir`` is used verbatim."""
    cfg = load_config(tmp_path / "absent.toml")
    cfg.paths.storage_dir = str(tmp_path / "abs storage")
    abs_persist = (tmp_path / "override" / "chroma").resolve()
    cfg.vec_index.persist_dir = str(abs_persist)

    paths = resolve_paths(cfg, project_root=tmp_path)
    assert paths.vec_index_dir == abs_persist


def test_vec_index_dir_default_under_storage(tmp_path):
    """Default ``persist_dir = "vec_index/chroma"`` lands at
    ``storage_dir/vec_index/chroma``."""
    cfg = load_config(tmp_path / "absent.toml")  # defaults
    paths = resolve_paths(cfg, project_root=tmp_path)
    assert paths.vec_index_dir == tmp_path / "storage" / "vec_index" / "chroma"
