---
name: prime-sleep
description: RETIRED alias — the sleep cycle (conversational extraction) is now the no-source branch of /prime-ingest. Invoking /prime-sleep just runs /prime-ingest with the same conversation scope. Triggers on "prime-sleep", "sleep cycle", "run a sleep cycle", "/prime-sleep 5", "/prime-sleep --all-pending".
---

`/prime-sleep` is **retired**. Conversational extraction (the sleep cycle) was unified
into **`/prime-ingest`** — it is now the branch that runs when no document/source flag is
given. The materialize rule, plan/extract/write steps, reconcile, and finalize are
identical; `/prime-ingest` just also handles document and conversation-source ingestion
in the same cycle.

**Do this:** run the **`/prime-ingest`** skill, passing through whatever conversation
scope the user gave:
- `/prime-sleep`            → `/prime-ingest`               (drain all pending)
- `/prime-sleep --all-pending` → `/prime-ingest --all-pending`
- `/prime-sleep 5`         → `/prime-ingest 5`             (limit 5)

The helper scripts in this directory (`plan.py`, `conv_extract.workflow.js`,
`writer.py`) are still used — `/prime-ingest`'s conversation branch invokes them by path.
Do not duplicate the steps here; follow `/prime-ingest`'s SKILL.md.
