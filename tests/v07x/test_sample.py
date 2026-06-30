"""v0.7-x W-E: deterministic size-stratified sample selector."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from priming_stream.cli.sample import (
    _cmd_sample_export,
    _total_chars,
    register,
    select_sample,
)


# -- fixtures -------------------------------------------------------------


def _make_conv(uuid: str, char_size: int) -> dict:
    """A minimal conversation with a single message of the requested size.

    Real exports have richer shape, but the selector cares only about
    ``chat_messages[*].text`` lengths.
    """
    return {
        "uuid": uuid,
        "name": f"conv-{uuid}",
        "created_at": "2026-01-01T00:00:00Z",
        "chat_messages": [
            {
                "sender": "human",
                "text": "x" * char_size,
                "created_at": "2026-01-01T00:00:00Z",
            }
        ],
    }


def _make_export(tmp_path: Path, sizes: list[int]) -> Path:
    """Write a ``conversations.json`` with conversations of the given sizes."""
    convs = [_make_conv(f"u{i}", size) for i, size in enumerate(sizes)]
    tmp_path.mkdir(parents=True, exist_ok=True)
    target = tmp_path / "conversations.json"
    target.write_text(
        json.dumps(convs, ensure_ascii=False), encoding="utf-8",
    )
    return target


# -- helpers --------------------------------------------------------------


def test_total_chars_sums_messages():
    conv = {
        "chat_messages": [
            {"text": "abc"},
            {"text": "defgh"},
            {"text": ""},
        ]
    }
    assert _total_chars(conv) == 8


def test_total_chars_empty_or_missing():
    assert _total_chars({}) == 0
    assert _total_chars({"chat_messages": None}) == 0
    assert _total_chars({"chat_messages": []}) == 0


# -- select_sample --------------------------------------------------------


def test_select_sample_picks_exactly_n():
    # 40 convs, sizes 1..40 thousand chars; should pick exactly 20.
    convs = [_make_conv(f"u{i}", (i + 1) * 1000) for i in range(40)]
    picked = select_sample(convs, n=20)
    assert len(picked) == 20


def test_select_sample_stratified_large_median_small():
    # 40 convs: sizes 1k, 2k, ..., 40k (ascending). uuids u0..u39.
    convs = [_make_conv(f"u{i}", (i + 1) * 1000) for i in range(40)]
    picked = select_sample(convs, n=20)
    uuids = {c["uuid"] for c in picked}

    # Top 5 by size are u35..u39 (sizes 36k..40k).
    for i in range(35, 40):
        assert f"u{i}" in uuids, f"missing large u{i}"

    # Bottom 5 (with total_chars > 1000): smallest eligible are u1..u5
    # (sizes 2k..6k; u0 at 1k is excluded by the >1000 filter).
    for i in range(1, 6):
        assert f"u{i}" in uuids, f"missing small u{i}"


def test_select_sample_excludes_tiny():
    """total_chars <= 1000 must not appear in the small stratum."""
    # 30 convs: 10 with 500 chars (empties), 20 with 5k chars.
    sizes = [500] * 10 + [5000] * 20
    convs = [_make_conv(f"u{i}", s) for i, s in enumerate(sizes)]
    picked = select_sample(convs, n=20)
    for c in picked:
        assert _total_chars(c) > 1000, f"tiny conv leaked in: {c['uuid']}"


def test_select_sample_deterministic():
    """Same input -> same output, byte-identical."""
    convs = [_make_conv(f"u{i}", (i + 1) * 1000) for i in range(40)]
    a = select_sample(convs, n=20)
    b = select_sample(convs, n=20)
    assert [c["uuid"] for c in a] == [c["uuid"] for c in b]


def test_select_sample_fewer_than_n_returns_all():
    convs = [_make_conv(f"u{i}", (i + 1) * 1000) for i in range(5)]
    picked = select_sample(convs, n=20)
    assert len(picked) == 5


def test_select_sample_empty_input():
    assert select_sample([], n=20) == []


def test_select_sample_zero_n():
    convs = [_make_conv("u0", 5000)]
    assert select_sample(convs, n=0) == []


# -- CLI ------------------------------------------------------------------


def test_register_adds_subcommand():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register(sub)
    # Must accept the new subcommand without error.
    args = parser.parse_args(
        ["sample-export", "--n", "20", "--in", "x", "--out", "y"]
    )
    assert args.n == 20
    assert args.in_path == "x"
    assert args.out == "y"


def test_cmd_sample_export_writes_subset(tmp_path):
    sizes = list(range(1000, 31000, 1000))  # 30 convs, 1k..30k chars
    in_path = _make_export(tmp_path / "in", sizes)
    out_dir = tmp_path / "out"
    args = argparse.Namespace(in_path=str(in_path), out=str(out_dir), n=20)
    rc = _cmd_sample_export(args)
    assert rc == 0
    out_file = out_dir / "conversations.json"
    assert out_file.is_file()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) == 20


def test_cmd_sample_export_accepts_directory_input(tmp_path):
    sizes = list(range(1000, 31000, 1000))
    _make_export(tmp_path / "in", sizes)
    out_dir = tmp_path / "out"
    args = argparse.Namespace(
        in_path=str(tmp_path / "in"), out=str(out_dir), n=20,
    )
    rc = _cmd_sample_export(args)
    assert rc == 0
    assert (out_dir / "conversations.json").is_file()


def test_cmd_sample_export_missing_input(tmp_path, capsys):
    args = argparse.Namespace(
        in_path=str(tmp_path / "nope.json"),
        out=str(tmp_path / "out"),
        n=20,
    )
    rc = _cmd_sample_export(args)
    assert rc == 1


def test_cmd_sample_export_warns_on_shortfall(tmp_path, capsys):
    sizes = [5000, 6000, 7000]
    _make_export(tmp_path / "in", sizes)
    args = argparse.Namespace(
        in_path=str(tmp_path / "in"),
        out=str(tmp_path / "out"),
        n=20,
    )
    rc = _cmd_sample_export(args)
    assert rc == 0
    err = capsys.readouterr().err
    assert "warn" in err.lower()
