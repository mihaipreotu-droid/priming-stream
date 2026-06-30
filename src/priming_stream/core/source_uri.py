"""Unified ``source_uri`` model for v0.7-x records.

Three schemes:

- ``qmd://<collection>/<relative-path>`` — materialized into the corpus dir
  (chunks live under ``imports/``, records under ``records/``).
  NOTE: the ``qmd://`` scheme name is a stable historical artifact. qmd
  was removed as a runtime dependency in v0.7-x-vec-index (replaced by
  in-process fastembed + ChromaDB), but the scheme is preserved verbatim
  on the 155 existing records' frontmatter and any new records keep
  emitting it. Renaming the scheme would be a frontmatter-breaking
  change and is explicitly out of scope.
- ``file:///<absolute-posix-path>`` — in-place project documents (§16.10).
  Path is rendered with forward slashes; on Windows the drive letter is
  the first path segment (``file:///C:/x/y.md``).
- ``claude_code_session://<session_id>`` — self-anchored records produced
  by slash commands (``/decizie``, ``/outcome``, ``/note``). No on-disk
  source — the record's summary IS the content. ``resolve`` returns
  ``None`` for these; callers handle the self-anchored case explicitly.

``parse`` / ``build`` round-trip. ``resolve`` maps a ``SourceURI`` back
to a concrete filesystem ``Path`` (or ``None`` for self-anchored URIs).
For ``qmd://`` URIs we mirror the on-disk layout under ``corpus_dir`` —
``priming-stream-imports`` → ``imports/`` and
``priming-stream-records`` → ``records/``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# NOTE: the literal authority strings ``priming-stream-imports`` and
# ``priming-stream-records`` are FROZEN URI authority names, NOT qmd
# collection names. They appear inside ``source_uri:`` frontmatter on
# the 155 existing records on disk; changing them would invalidate that
# frontmatter (frontmatter shape is explicitly out of scope per the
# v0.7-x-vec-index brief). Treat these strings as opaque historical
# tokens, like a database row-id schema you inherited.
_AUTHORITY_IMPORTS = "priming-stream-imports"
_AUTHORITY_RECORDS = "priming-stream-records"

# Local-subdir mapping for the two Priming Stream URI authorities. Other authorities
# (e.g. user-configured in-place doc collections) are not corpus-resident —
# ``resolve`` rejects them since the caller must provide the in-place path
# explicitly via config rather than guessing from authority name.
_COLLECTION_SUBDIRS = {
    _AUTHORITY_IMPORTS: "imports",
    _AUTHORITY_RECORDS: "records",
}


@dataclass(frozen=True)
class SourceURI:
    scheme: str                          # 'qmd', 'file', or 'claude_code_session'
    collection: str | None               # collection name (qmd only); None for
                                         # file / claude_code_session
    path: str                            # qmd: relative inside collection (forward
                                         # slashes, no leading slash);
                                         # file: absolute posix path
                                         # (with leading slash on POSIX; on
                                         # Windows the first segment is the
                                         # drive letter, e.g. 'C:/x/y.md');
                                         # claude_code_session: the session id


def parse(s: str) -> SourceURI:
    """Parse a ``qmd://``, ``file:///`` or ``claude_code_session://`` URI."""
    if s.startswith("qmd://"):
        rest = s[len("qmd://"):]
        if "/" not in rest:
            raise ValueError(f"qmd URI missing path: {s!r}")
        collection, _, path = rest.partition("/")
        if not collection:
            raise ValueError(f"qmd URI missing collection: {s!r}")
        return SourceURI(scheme="qmd", collection=collection, path=path)
    if s.startswith("file:///"):
        # 'file:///C:/x/y.md' -> path = 'C:/x/y.md'
        # 'file:///abs/path'  -> path = 'abs/path'  (POSIX)
        path = s[len("file:///"):]
        return SourceURI(scheme="file", collection=None, path=path)
    if s.startswith("claude_code_session://"):
        # Self-anchored records (slash commands) — path is the session id.
        path = s[len("claude_code_session://"):]
        return SourceURI(scheme="claude_code_session", collection=None, path=path)
    raise ValueError(f"unsupported URI scheme: {s!r}")


def build(scheme: str, *, collection: str | None = None, path: str) -> str:
    """Build a URI string from components. Inverse of ``parse``."""
    if scheme == "qmd":
        if not collection:
            raise ValueError("qmd scheme requires a collection name")
        norm = path.replace("\\", "/").lstrip("/")
        return f"qmd://{collection}/{norm}"
    if scheme == "file":
        if collection is not None:
            raise ValueError("file scheme does not take a collection")
        # Path is expected absolute posix-style. Strip a single leading
        # slash if present (we re-prepend it in the 'file:///' literal).
        norm = path.replace("\\", "/")
        norm = norm.lstrip("/")
        return f"file:///{norm}"
    if scheme == "claude_code_session":
        if collection is not None:
            raise ValueError(
                "claude_code_session scheme does not take a collection"
            )
        return f"claude_code_session://{path}"
    raise ValueError(f"unsupported scheme: {scheme!r}")


def resolve(
    uri: SourceURI, storage_dir: Path, corpus_dir: Path,
) -> Path | None:
    """Resolve a ``SourceURI`` to a concrete filesystem path.

    - ``qmd://priming-stream-imports/<p>``  -> ``corpus_dir / 'imports' / p``
    - ``qmd://priming-stream-records/<p>``  -> ``corpus_dir / 'records' / p``
    - ``file:///<abs>``                        -> ``Path(<abs>)``
    - ``claude_code_session://<sid>``          -> ``None`` (self-anchored)

    ``storage_dir`` is currently unused (the corpus root is supplied
    explicitly), but kept in the signature for symmetry with other path
    resolvers in the codebase and to leave room for ``file:///`` URIs that
    might one day be expressed relative to the project root.

    The ``corpus_dir`` parameter is the new (v0.7-x-vec-index) name for
    what used to be ``qmd_corpus_dir`` — see module docstring on why the
    ``qmd://`` scheme name is preserved despite the rename.

    Unknown collection -> ``ValueError``.
    """
    _ = storage_dir  # reserved
    if uri.scheme == "qmd":
        sub = _COLLECTION_SUBDIRS.get(uri.collection or "")
        if sub is None:
            raise ValueError(
                f"unknown qmd collection for resolve(): {uri.collection!r}"
            )
        return corpus_dir / sub / uri.path
    if uri.scheme == "file":
        return Path(uri.path)
    if uri.scheme == "claude_code_session":
        # Self-anchored: the record's summary IS the source. No on-disk path.
        return None
    raise ValueError(f"unsupported scheme: {uri.scheme!r}")
