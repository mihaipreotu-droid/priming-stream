# Priming Stream — Setup

End-to-end install for the Priming Stream POC. Single-user; developed and tested on Windows (the code is
cross-platform Python — only the optional scheduler is Windows-specific).
Every step is idempotent: re-running an install command never duplicates entries
and never removes third-party ones.

## 1. Prerequisites

- Python 3.12 or newer.
- [Claude Code](https://docs.claude.com/claude-code) installed and authenticated
  via your Claude Pro/Max subscription. The sleep cycle's judgment steps
  (extraction, reconciliation) run as Claude Code **Workflow** agents against your
  subscription quota — no separate API key needed.
- Claude Desktop (optional) — if you want the desktop MCP integration.
- An OS. Developed and tested on Windows 10 / 11; the rest is cross-platform Python.
  The unattended scheduler step uses `schtasks` (Windows); on Linux/macOS the scheduler
  and Windows-specific `doctor` checks soft-skip, and you'd wire `scripts/sleep_auto.cmd`'s
  command into cron yourself.
- For the **optional unattended** sleep cycle only: an OAuth token from
  `claude setup-token` (opens a browser, authorizes your Claude account, produces a
  long-lived `sk-ant-oat01-…` token). Set it at User scope as
  `CLAUDE_CODE_OAUTH_TOKEN` so the scheduled task inherits it.

## 2. Quick start

From the repository root:

```powershell
pip install -e .
prime init
prime install-hooks
prime install-mcp --client both
prime doctor
```

The sequence sets up storage, wires the four waking-time hooks, registers the
MCP server with both Claude Code and Claude Desktop, and runs all health checks.
A clean run ends with every `doctor` line reading `[PASS]` or `[SKIP]`. For the
optional unattended nightly consolidation, see §3 "Scheduling".

## 3. Per-command details

### `prime init`

Creates `storage/`, `storage/episodic/`, `storage/corpus/`, and an empty
`storage/graph.db` with the schema applied. Idempotent — safe to re-run;
`apply_migrations` uses `CREATE TABLE IF NOT EXISTS`.

### `prime install-hooks`

Writes to `.claude/settings.json` (CWD-relative). Installs four hooks pointing
at `python -m priming_stream.hooks.<event>`:

- `UserPromptSubmit`
- `SessionStart`
- `Stop`
- `SessionEnd`

Reads any existing file, preserves all unrelated keys, replaces only the
Priming Stream entries (recognized by the substring `priming_stream.hooks` in the command).
One status line printed per event: `added`, `updated`, or `unchanged`.

### `prime install-mcp --client {claude_code|claude_desktop|both}`

Registers a `priming-stream` MCP server entry:

- `claude_code` → `.mcp.json` at CWD.
- `claude_desktop` → `%APPDATA%\Claude\claude_desktop_config.json` on Windows.
  For tests and overrides, set the `PRIMING_STREAM_DESKTOP_CONFIG` env var to a
  custom path.
- `both` → does both in one call.

Other `mcpServers` entries pass through untouched.

### `prime doctor`

Runs every check and prints one line per check:

1. Storage initialized (`storage/graph.db` exists).
2. Schema migration current (`records` table present after `apply_migrations`).
3. Hooks installed (4 events reference `priming_stream.hooks`).
4. MCP entry present (in `.mcp.json` or Claude Desktop config).
5. MCP server importable (`import priming_stream.mcp_server.server` succeeds).

Exit code 0 if every check passes or skips; 1 if any check fails.

### Scheduling the unattended sleep cycle (optional)

Consolidation can run unattended via the Windows Task Scheduler. The repo ships a
ready task definition; register it once (after setting the token from §1):

```powershell
schtasks /Create /TN PrimingStreamSleepAuto /XML scripts\sleep_auto_task.xml
```

The task runs `scripts\sleep_auto.cmd` (`prime sleep-auto --limit 40`) daily at
03:00, with catch-up if the machine was off. It inherits `CLAUDE_CODE_OAUTH_TOKEN`
from the User environment and only runs while you're logged in. Edit the XML to
change the cadence. On Linux/macOS, schedule `scripts/sleep_auto.cmd`'s command
with your own cron.

## 4. Manual fallback

If you'd rather wire things by hand (or an install step fails):

### `.claude/settings.json` (hooks)

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {"hooks": [{"type": "command", "command": "python -m priming_stream.hooks.user_prompt_submit"}]}
    ],
    "SessionStart": [
      {"hooks": [{"type": "command", "command": "python -m priming_stream.hooks.session_start"}]}
    ],
    "Stop": [
      {"hooks": [{"type": "command", "command": "python -m priming_stream.hooks.stop"}]}
    ],
    "SessionEnd": [
      {"hooks": [{"type": "command", "command": "python -m priming_stream.hooks.session_end"}]}
    ]
  }
}
```

### `.mcp.json` (Claude Code MCP server)

```json
{
  "mcpServers": {
    "priming-stream": {
      "command": "python",
      "args": ["-m", "priming_stream.mcp_server.server"]
    }
  }
}
```

### `%APPDATA%\Claude\claude_desktop_config.json` (Claude Desktop MCP server)

Same shape as `.mcp.json`. Full path on Windows is typically
`C:\Users\<you>\AppData\Roaming\Claude\claude_desktop_config.json`.

### Scheduled task (manual)

```cmd
schtasks /Create /TN PrimingStreamSleepAuto /XML scripts\sleep_auto_task.xml
```

The shipped XML runs `scripts\sleep_auto.cmd` daily; re-running with the same
`/TN` replaces the task in place.

## 5. Claude Desktop

Claude Desktop participates through two channels:

- **MCP query baseline** — the registered `priming-stream` MCP server gives
  Desktop read-only access to the same substrate.
- **Opt-in pull-bridge** — Desktop sessions can request bridge injection via
  the MCP tools rather than waking-time auto-injection.

Copy `docs/desktop-project-instruction.md` into the project's Custom
Instructions to opt into the pull-bridge (`graph_salient_context`).

### Disambiguation tool — Claude Code projects

Claude Code already gets live priming via the `UserPromptSubmit` hook —
no Custom Instructions needed for the baseline. To also enable the
**disambiguation tool** for ambiguous prompts (pronouns, deictics,
paraphrases), paste this snippet into the project's `CLAUDE.md` (project
constitution) — project-level so it stays scoped to where you want it:

```markdown
## Disambiguation tool — graph_disambiguate

When the user's prompt uses ambiguous references — pronouns ("aia, asta,
ăla"), deictics ("săptămâna trecută, ieri, recent"), vague nouns ("chestia
aia, lucrul de"), or paraphrases of earlier discussion — call the
`graph_disambiguate` MCP tool with your best canonical reformulation in
the `text` argument. The tool runs the live bridge against your reformulation
and returns the salient-context markdown. Read it the same way as the
priming the hook injects automatically.

Do NOT call when:
- The prompt is an acknowledgement or closing ("mersi", "ok", "thanks").
- The object is named explicitly (e.g. "what is PageRank?") — the named
  object is already a seed for the live bridge.
- You just called it on the previous turn for the same reformulation.

If the live bridge prepends a "Note from the Priming Stream" with a
shortage flag, treat that as an explicit invitation to call this tool.
```

Per-project so unrelated projects don't pull it in. The Priming Stream itself
doesn't auto-edit your `CLAUDE.md` — copy this when you want it.

## 6. Coldstart

For seeding the substrate from your existing claude.ai conversation exports,
see `prime coldstart`. Point a TOML manifest (see `coldstart-example.toml`)
at one or more export folders; `coldstart --config <file>` materializes those
conversations into the episodic log, so a following `/prime-ingest` sleep cycle
has something to extract from — giving a fresh install a non-empty substrate
to bridge against.

## 7. Troubleshooting / Doctor

`prime doctor` is the first thing to run when something's off.

| Check | Failure means | Fix |
|---|---|---|
| Storage initialized | No `storage/graph.db` | Run `prime init`. |
| Schema migration current | Table missing or DB corrupt | Back up DB, run `prime init` (re-applies migrations). |
| Hooks installed | `.claude/settings.json` missing or no Priming Stream entries | Run `prime install-hooks`. |
| MCP entry present | Neither `.mcp.json` nor Desktop config registers the server | Run `prime install-mcp --client both`. |
| MCP server importable | Package install broken | Re-run `pip install -e .`. |

Doctor returns 0 if every check passes.

### Stale editable install (`import priming_stream` resolves outside the repo)

`pip install -e .` writes an `__editable__.priming_stream-*.pth` into the
active Python's `site-packages` pointing at this repo's `src/`. If you
re-clone the repo, move it, or switch to a different worktree/branch checkout,
that `.pth` can keep pointing at the **old** location — so `import priming_stream`
silently resolves to stale code: tests pass against the wrong tree, the daemon
serves an old substrate, and edits appear to have no effect.

Check where `priming_stream` actually loads from:

```powershell
python -c "import priming_stream; print(priming_stream.__file__)"
```

If the printed path is **not** under the repo you're working in, re-pin the
editable install from the repo root:

```powershell
pip install -e .
```

This rewrites the `.pth` to the current `src/`. Re-run `prime doctor` to
confirm `MCP server importable` still passes.
