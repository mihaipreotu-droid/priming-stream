# Priming Stream

> *Associative memory for AI agents.*

A persistent, **owner-controlled associative memory** for an AI agent working in
[Claude Code](https://docs.claude.com/claude-code). It gives the assistant a durable, cross-session
*priming* layer: as you work, past conversations and documents are distilled into compact memory
**records**, embedded in a vector space, and — on each new prompt — a multi-hop *spreading-activation*
walk surfaces the records most associatively related to what you're doing, injected as context.

It is **not** a personal knowledge base, an autocomplete, or a claim that the model has a mind. Think
of it as an **external, System-1-*like* substrate**: a *prior to verify*, not ground truth — a weighted
field of associations surfaced so the live reasoning (the model's "System 2") can use or ignore it. Three
things set it apart from typical agent memory: priming is **automatic and unconditional** — surfaced as
background context on *every* turn, not fetched on demand the way query-driven memory systems work; it is
**owner-controlled and inspectable** — you read, edit, and retract what it holds, and records point back
to their source to verify (most agent memory is opaque); and it **persists across sessions and
consolidates offline** — a dedicated *sleep cycle* dedups and supersedes records so the substrate stays
coherent instead of growing into noise, externalizing the associative recall the model otherwise loses
when its context resets each session.

Retrieval is by **spreading activation** — itself an uncommon choice (most agent memory is plain vector
or keyword lookup), so worth knowing about as a feature; but a few contemporary cognitive-memory systems
share it, so it's a capability rather than the differentiator. The distinctive part is the passive,
every-turn priming around it.

## Status

Proof-of-concept. **Single-user, owner-controlled, Windows-first, Claude-bound** (it runs under Claude
Code and uses your Claude subscription for the offline distillation step — no third-party API, no local
models). Developed and tested on Windows; the code is cross-platform Python and only the unattended
scheduler is Windows-specific (it soft-skips elsewhere, with a cron alternative) — Mac/Linux are
untested. Not production-hardened; no multi-user, cloud sync, or mobile.

## How it works (one paragraph)

Two memory systems. **Episodic**: conversation transcripts and ingested documents are chunked and kept
immutable on disk. **Semantic**: durable *records* — `claim`s distilled from conversations and
`index_card`s for documents — stored in a canonical SQLite database (`graph.db`) and embedded locally
(`fastembed` + ChromaDB, derived and rebuildable). The "graph" is **implicit, in embedding space**; the
bridge that primes a live turn is a multi-hop spreading walk over it (mathematically a
random-walk-with-restart / personalized-PageRank over implicit similarity edges). Writing the durable
layer happens **only** in an offline "sleep cycle" (batched extraction + reconciliation), never per
turn; reading and injecting happen live in a lightweight hook.

## Install

See [SETUP.md](SETUP.md) for the full, idempotent install (Python 3.12+, Claude Code, an OAuth token
from `claude setup-token`). Short version:

```powershell
pip install -e .
prime init
prime install-hooks
prime install-mcp --client both
prime doctor
```

## Usage (sketch)

- **Priming is automatic.** Once hooks are installed, each prompt in a Claude Code project gets a
  "Salient context — memory records" block injected by the `UserPromptSubmit` hook.
- **Feed the substrate.** `/prime-ingest` ingests conversations and documents and runs a sleep cycle
  (extract → reconcile → finalize). A scheduled task can run it unattended on idle.
- **Curate by hand.** `prime record create | edit | delete` for owner-authored memories. Writes go
  through the CLI; the MCP server is strictly **read-only**.
- **Inspect.** `prime dashboard`, `prime echoes`, `prime search` to see what's in the substrate
  and what gets primed.

## A note on framing

The neuroscience vocabulary here (System 1, spreading activation, priming, consolidation, sleep) is
used as **analogy and design lineage** (the architecture sits on the MINERVA-2 / multiple-trace and
spreading-activation literature), not as a claim of biological equivalence or that the AI possesses a
real System 1. `source_date` is when the *conversation* happened, not when a fact became true — treat
surfaced records as a lossy *pointer* to their source, a prior to verify, not evidence.

## License

[MIT](LICENSE).
