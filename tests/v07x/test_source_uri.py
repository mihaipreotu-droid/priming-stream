"""v0.7-x W-B: SourceURI parse/build round-trip + resolve."""
from __future__ import annotations

from pathlib import Path

import pytest

from priming_stream.core.source_uri import SourceURI, build, parse, resolve


# -- parse / build round-trip --------------------------------------------


def test_parse_qmd():
    uri = parse("qmd://priming-stream-imports/foo/bar.md")
    assert uri == SourceURI(
        scheme="qmd",
        collection="priming-stream-imports",
        path="foo/bar.md",
    )


def test_parse_file_windows_path():
    uri = parse("file:///C:/x/y.md")
    assert uri == SourceURI(scheme="file", collection=None, path="C:/x/y.md")


def test_parse_file_posix_path():
    uri = parse("file:///home/user/x.md")
    assert uri == SourceURI(
        scheme="file", collection=None, path="home/user/x.md"
    )


def test_build_qmd_round_trip():
    s = build("qmd", collection="priming-stream-imports", path="x/y.md")
    assert s == "qmd://priming-stream-imports/x/y.md"
    assert parse(s) == SourceURI(
        scheme="qmd",
        collection="priming-stream-imports",
        path="x/y.md",
    )


def test_build_file_round_trip_windows():
    s = build("file", path="C:/x/y.md")
    assert s == "file:///C:/x/y.md"
    assert parse(s).path == "C:/x/y.md"


def test_build_normalizes_backslashes():
    # Path coming in with Windows separators is normalized to forward
    # slashes in the URI form (POSIX inside source_uri — convention §10).
    s = build("qmd", collection="ps-imports", path="a\\b\\c.md")
    assert s == "qmd://ps-imports/a/b/c.md"


def test_build_strips_leading_slash_on_qmd_path():
    s = build("qmd", collection="ps-imports", path="/a/b.md")
    assert s == "qmd://ps-imports/a/b.md"


# -- error paths ---------------------------------------------------------


def test_parse_rejects_unknown_scheme():
    with pytest.raises(ValueError):
        parse("http://example.com/x.md")


def test_parse_rejects_qmd_missing_collection():
    with pytest.raises(ValueError):
        parse("qmd:///foo/bar.md")


def test_build_qmd_requires_collection():
    with pytest.raises(ValueError):
        build("qmd", path="x.md")


def test_build_file_rejects_collection():
    with pytest.raises(ValueError):
        build("file", collection="x", path="y.md")


# -- resolve --------------------------------------------------------------


def test_resolve_qmd_imports(tmp_path):
    uri = parse("qmd://priming-stream-imports/claude_ai_export/sess/c0.md")
    corpus = tmp_path / "corpus"
    out = resolve(uri, storage_dir=tmp_path, corpus_dir=corpus)
    assert out == corpus / "imports" / "claude_ai_export" / "sess" / "c0.md"


def test_resolve_qmd_records(tmp_path):
    uri = parse("qmd://priming-stream-records/rec_abc12345.md")
    corpus = tmp_path / "corpus"
    out = resolve(uri, storage_dir=tmp_path, corpus_dir=corpus)
    assert out == corpus / "records" / "rec_abc12345.md"


def test_resolve_file(tmp_path):
    uri = parse("file:///C:/x/y.md")
    out = resolve(uri, storage_dir=tmp_path, corpus_dir=tmp_path)
    # POSIX-style C:/x/y.md back to a Path on the running platform.
    assert Path("C:/x/y.md") == out


def test_resolve_unknown_collection_raises(tmp_path):
    uri = SourceURI(scheme="qmd", collection="some-other-coll", path="x.md")
    with pytest.raises(ValueError):
        resolve(uri, storage_dir=tmp_path, corpus_dir=tmp_path)


# -- claude_code_session scheme (F-2) ------------------------------------


def test_parse_claude_code_session_uri():
    uri = parse("claude_code_session://sess-1")
    assert uri == SourceURI(
        scheme="claude_code_session", collection=None, path="sess-1",
    )


def test_build_claude_code_session_round_trip():
    s = build("claude_code_session", path="sess-1")
    assert s == "claude_code_session://sess-1"
    assert parse(s) == SourceURI(
        scheme="claude_code_session", collection=None, path="sess-1",
    )


def test_build_claude_code_session_rejects_collection():
    with pytest.raises(ValueError):
        build("claude_code_session", collection="x", path="sess-1")


def test_resolve_claude_code_session_returns_none(tmp_path):
    uri = parse("claude_code_session://sess-1")
    out = resolve(uri, storage_dir=tmp_path, corpus_dir=tmp_path)
    assert out is None


def test_parse_claude_code_session_unknown_session_path():
    # The 'unknown' path is the literal session-id sentinel used by the
    # bridge when session_id is absent (see bridge.commands).
    uri = parse("claude_code_session://unknown")
    assert uri.scheme == "claude_code_session"
    assert uri.path == "unknown"
