"""R5 — ``cmd_sleep_finalize`` calls ``reload_daemon`` at cycle end.

Mocks ``priming_stream.daemon.client.reload_daemon`` and asserts the sleep-finalize
exit code, stdout/stderr content under three branches:

* R5.a — daemon running (mock returns dict): reload line on stdout, rc 0.
* R5.b — daemon NOT running (mock returns None): no reload line, rc 0.
* R5.c — reload raises: stderr message, rc 0 (records are durable).
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pytest

from priming_stream.cli import sleep as sleep_cli
from priming_stream.core.db import connect
from priming_stream.core.graph_repo import GraphRepo
from priming_stream.core.schema import apply_migrations


# ---- shared fixtures (mirror test_sleep_cli) ---------------------------


@pytest.fixture
def project(tmp_path, monkeypatch):
    project_root = tmp_path / "proj"
    project_root.mkdir()
    monkeypatch.chdir(project_root)
    storage = project_root / "storage"
    storage.mkdir()
    db = storage / "graph.db"
    conn = connect(db)
    try:
        apply_migrations(conn)
    finally:
        conn.close()
    return project_root


def _open_cycle(project_root: Path) -> int:
    conn = connect(project_root / "storage" / "graph.db")
    try:
        return GraphRepo(conn).start_sleep_cycle(
            started_at="2026-05-27T10:00:00Z",
        )
    finally:
        conn.close()


def _finalize_args(cycle_id: int) -> argparse.Namespace:
    return argparse.Namespace(
        cycle_id=cycle_id,
        chunks_materialized=0,
        records_created=0,
        records_skipped=0,
        notes=None,
        skip_vec_index=True,
    )


# ---- R5.a — daemon running prints reload line -------------------------


def test_r5a_prints_reload_line_on_daemon_running(project):
    cycle_id = _open_cycle(project)
    canned = {
        "status": "ok",
        "reload_ms": 84.2,
        "records_before": 155,
        "records_after": 165,
        "daemon_version": "v0.7-x-bridge-daemon",
    }

    out = io.StringIO()
    err = io.StringIO()
    with patch(
        "priming_stream.daemon.client.reload_daemon", return_value=canned,
    ) as mock_reload:
        with redirect_stdout(out), redirect_stderr(err):
            rc = sleep_cli.cmd_sleep_finalize(_finalize_args(cycle_id))

    assert rc == 0
    assert mock_reload.call_count == 1
    # Timeout passed through (spec §4.4).
    _, kwargs = mock_reload.call_args
    assert kwargs.get("timeout_s") == 5.0

    stdout = out.getvalue()
    assert "[sleep-finalize] daemon reloaded:" in stdout
    assert "155 -> 165 records" in stdout
    assert "84ms" in stdout
    assert err.getvalue() == ""


# ---- R5.b — daemon NOT running, silent skip ----------------------------


def test_r5b_silent_skip_when_daemon_not_running(project):
    cycle_id = _open_cycle(project)

    out = io.StringIO()
    err = io.StringIO()
    with patch(
        "priming_stream.daemon.client.reload_daemon", return_value=None,
    ) as mock_reload:
        with redirect_stdout(out), redirect_stderr(err):
            rc = sleep_cli.cmd_sleep_finalize(_finalize_args(cycle_id))

    assert rc == 0
    assert mock_reload.call_count == 1
    stdout = out.getvalue()
    # The JSON manifest still printed; no reload-line.
    assert "[sleep-finalize] daemon reloaded" not in stdout
    assert "[sleep-finalize] daemon reload failed" not in stdout
    assert err.getvalue() == ""


# ---- R5.c — reload raises, stderr message, rc 0 ------------------------


def test_r5c_reload_failure_is_non_fatal(project):
    cycle_id = _open_cycle(project)

    out = io.StringIO()
    err = io.StringIO()
    with patch(
        "priming_stream.daemon.client.reload_daemon",
        side_effect=RuntimeError("reload returned 500: boom"),
    ) as mock_reload:
        with redirect_stdout(out), redirect_stderr(err):
            rc = sleep_cli.cmd_sleep_finalize(_finalize_args(cycle_id))

    assert rc == 0  # non-fatal
    assert mock_reload.call_count == 1
    assert "[sleep-finalize] daemon reload failed:" in err.getvalue()
    assert "reload returned 500: boom" in err.getvalue()
    # JSON manifest still printed before the reload attempt.
    assert json.loads(out.getvalue())["cycle_id"] == cycle_id


# ---- cycle still closes on the SQLite side ------------------------------


def test_sleep_finalize_closes_row_even_when_reload_fails(project):
    cycle_id = _open_cycle(project)
    with patch(
        "priming_stream.daemon.client.reload_daemon",
        side_effect=RuntimeError("transport error"),
    ):
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = sleep_cli.cmd_sleep_finalize(_finalize_args(cycle_id))
    assert rc == 0

    conn = connect(project / "storage" / "graph.db")
    try:
        cycles = GraphRepo(conn).list_sleep_cycles()
    finally:
        conn.close()
    row = next(c for c in cycles if c["id"] == cycle_id)
    assert row["completed_at"] is not None
