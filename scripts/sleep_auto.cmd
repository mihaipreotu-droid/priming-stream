@echo off
REM Priming Stream — unattended nightly consolidation (W automation).
REM Run by Windows Task Scheduler. Inherits CLAUDE_CODE_OAUTH_TOKEN from the
REM user environment (set once via: setx CLAUDE_CODE_OAUTH_TOKEN "<token>").
REM
REM cwd MUST be the Priming Stream project dir so resolve_paths resolves the real
REM storage (the workflow scripts resolve it via `git rev-parse --show-toplevel`).
REM All projects are included —
REM the extraction worker filters relevance; sub-agent transcripts are excluded
REM structurally (not conversations). Add --limit N to cap a large backlog.

REM %~dp0 is this script's dir (<repo>\scripts\); ".." is the repo root.
cd /d "%~dp0.."
python -m priming_stream.cli.main sleep-auto --limit 40
