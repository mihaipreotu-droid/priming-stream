"""Install + wiring + doctor subcommands (v0.7-x).

Wires the Priming Stream into Claude Code / Claude Desktop. v0.7-x dropped the
``install-scheduler`` subcommand: consolidation runs via the ``/prime-ingest``
Claude Code skill (or headless via ``prime sleep-auto`` +
``scripts/sleep_auto.cmd`` on Windows Task Scheduler), not as a scheduled
``prime sleep`` subprocess.

Paths are resolved relative to the current working directory (``.claude/``,
``.mcp.json``). The Claude Desktop config path is platform-conventional on
Windows (``%APPDATA%/Claude/claude_desktop_config.json``) and can be
overridden via the ``PRIMING_STREAM_DESKTOP_CONFIG`` env var for tests.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from priming_stream.core.config import load_config
from priming_stream.core.paths import resolve_paths


# -- shared constants ----------------------------------------------------

HOOK_EVENTS: dict[str, str] = {
    "UserPromptSubmit": "python -m priming_stream.hooks.user_prompt_submit",
    "SessionStart": "python -m priming_stream.hooks.session_start",
    "Stop": "python -m priming_stream.hooks.stop",
    "SessionEnd": "python -m priming_stream.hooks.session_end",
}

MCP_SERVER_KEY = "priming-stream"
MCP_SERVER_ENTRY = {
    "command": "python",
    "args": ["-m", "priming_stream.mcp_server.server"],
}


# -- helpers --------------------------------------------------------------


def _read_json(path: Path) -> dict:
    """Read JSON if present, else {}. Malformed JSON also yields {}."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _desktop_config_path() -> Path:
    """Resolve the Claude Desktop config path (Windows convention or env)."""
    override = os.environ.get("PRIMING_STREAM_DESKTOP_CONFIG")
    if override:
        return Path(override)
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "Claude" / "claude_desktop_config.json"
    return Path.home() / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json"


# -- install-hooks --------------------------------------------------------


def _entry_for(command: str) -> dict:
    return {"hooks": [{"type": "command", "command": command}]}


def _merge_hook_event(
    existing: list, command: str,
) -> tuple[list, str]:
    """Return (new_list, status) where status is added|updated|unchanged.

    Recognizes a Priming Stream entry by the substring ``priming_stream.hooks`` anywhere
    inside the serialized entry. Third-party entries pass through untouched.
    """
    out: list = []
    status = "added"
    found = False
    for entry in existing:
        serialized = json.dumps(entry)
        if "priming_stream.hooks" in serialized:
            found = True
            new_entry = _entry_for(command)
            if entry == new_entry:
                status = "unchanged"
            else:
                status = "updated"
            out.append(new_entry)
        else:
            out.append(entry)
    if not found:
        out.append(_entry_for(command))
    return out, status


def install_hooks(args: argparse.Namespace) -> int:
    settings_path = Path.cwd() / ".claude" / "settings.json"
    data = _read_json(settings_path)
    hooks = data.get("hooks")
    if not isinstance(hooks, dict):
        hooks = {}

    for event, command in HOOK_EVENTS.items():
        existing = hooks.get(event)
        if not isinstance(existing, list):
            existing = []
        merged, status = _merge_hook_event(existing, command)
        hooks[event] = merged
        print(f"[install-hooks] {event}: {status}")

    data["hooks"] = hooks
    _write_json(settings_path, data)
    return 0


# -- install-mcp ----------------------------------------------------------


def _install_mcp_to(path: Path, label: str) -> None:
    data = _read_json(path)
    servers = data.get("mcpServers")
    if not isinstance(servers, dict):
        servers = {}

    if MCP_SERVER_KEY in servers:
        if servers[MCP_SERVER_KEY] == MCP_SERVER_ENTRY:
            status = "unchanged"
        else:
            status = "updated"
    else:
        status = "added"
    servers[MCP_SERVER_KEY] = MCP_SERVER_ENTRY
    data["mcpServers"] = servers
    _write_json(path, data)
    print(f"[install-mcp] {label} ({path}): {status}")


def install_mcp(args: argparse.Namespace) -> int:
    client = args.client
    if client in ("claude_code", "both"):
        _install_mcp_to(Path.cwd() / ".mcp.json", "claude_code")
    if client in ("claude_desktop", "both"):
        _install_mcp_to(_desktop_config_path(), "claude_desktop")
    return 0


# -- doctor ---------------------------------------------------------------


def _check_storage(report: list[str]) -> bool:
    cfg = load_config()
    paths = resolve_paths(cfg, project_root=Path.cwd())
    if paths.graph_db.exists():
        report.append(f"[PASS] storage initialized ({paths.graph_db})")
        return True
    report.append(
        f"[FAIL] storage initialized: graph.db missing at {paths.graph_db}"
    )
    return False


def _check_schema(report: list[str]) -> bool:
    from priming_stream.core.db import connect
    from priming_stream.core.schema import apply_migrations

    cfg = load_config()
    paths = resolve_paths(cfg, project_root=Path.cwd())
    if not paths.graph_db.exists():
        report.append("[FAIL] schema migration: no graph.db to migrate")
        return False
    try:
        conn = connect(paths.graph_db)
        try:
            apply_migrations(conn)
            cur = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='records'"
            )
            present = cur.fetchone() is not None
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 — doctor reports, never crashes
        report.append(f"[FAIL] schema migration: {exc}")
        return False
    if present:
        report.append("[PASS] schema migration current")
        return True
    report.append("[FAIL] schema migration: 'records' table missing")
    return False


def _check_hooks(report: list[str]) -> bool:
    settings = Path.cwd() / ".claude" / "settings.json"
    if not settings.exists():
        report.append(f"[FAIL] hooks installed: {settings} not found")
        return False
    data = _read_json(settings)
    hooks = data.get("hooks", {})
    missing = []
    for event in HOOK_EVENTS:
        entries = hooks.get(event) or []
        flat = json.dumps(entries)
        if "priming_stream.hooks" not in flat:
            missing.append(event)
    if missing:
        report.append(
            f"[FAIL] hooks installed: missing Priming Stream entry for {missing}"
        )
        return False
    report.append("[PASS] hooks installed")
    return True


def _check_mcp(report: list[str]) -> bool:
    candidates = [Path.cwd() / ".mcp.json", _desktop_config_path()]
    for candidate in candidates:
        if not candidate.exists():
            continue
        data = _read_json(candidate)
        servers = data.get("mcpServers", {}) or {}
        if MCP_SERVER_KEY in servers:
            report.append(f"[PASS] MCP entry present ({candidate})")
            return True
    report.append(
        "[FAIL] MCP entry: no 'priming-stream' server in .mcp.json or "
        "Claude Desktop config"
    )
    return False


def _check_mcp_importable(report: list[str]) -> bool:
    try:
        import importlib

        importlib.import_module("priming_stream.mcp_server.server")
    except Exception as exc:  # noqa: BLE001
        report.append(f"[FAIL] MCP server importable: {exc}")
        return False
    report.append("[PASS] MCP server importable")
    return True


def doctor(args: argparse.Namespace) -> int:
    report: list[str] = []
    checks = [
        _check_storage,
        _check_schema,
        _check_hooks,
        _check_mcp,
        _check_mcp_importable,
    ]
    all_ok = True
    for check in checks:
        try:
            ok = check(report)
        except Exception as exc:  # noqa: BLE001 — doctor must keep going
            report.append(f"[FAIL] {check.__name__}: unexpected error: {exc}")
            ok = False
        if not ok:
            all_ok = False
    for line in report:
        print(line)
    return 0 if all_ok else 1


# -- registration ---------------------------------------------------------


def register(subparsers) -> None:
    """Attach install-hooks, install-mcp, doctor."""
    p_hooks = subparsers.add_parser(
        "install-hooks",
        help="install Claude Code hooks into .claude/settings.json",
    )
    p_hooks.set_defaults(func=install_hooks)

    p_mcp = subparsers.add_parser(
        "install-mcp",
        help="register the priming-stream MCP server",
    )
    p_mcp.add_argument(
        "--client",
        choices=["claude_code", "claude_desktop", "both"],
        default="claude_code",
    )
    p_mcp.set_defaults(func=install_mcp)

    p_doctor = subparsers.add_parser(
        "doctor",
        help="diagnose Priming Stream install and runtime",
    )
    p_doctor.set_defaults(func=doctor)
