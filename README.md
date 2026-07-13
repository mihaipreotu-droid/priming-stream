# Priming Stream

> *Associative memory for AI agents.*

[![CI](https://github.com/mihaipreotu-droid/priming-stream/actions/workflows/ci.yml/badge.svg)](https://github.com/mihaipreotu-droid/priming-stream/actions/workflows/ci.yml)

Human memory is associative and spontaneous. As you perceive or reason, related memories — episodic and
semantic — surface on their own: some tied closely to the matter at hand, some far from it. That mechanism
underlies not just remembering but *thinking* itself; creative thought most of all is the making of
connections that are unexpected and distant, yet meaningful. **Priming Stream** sets out to give an AI agent
working in [Claude Code](https://docs.claude.com/claude-code) something of that capacity.

Two properties make it practical to run for real:

- **Cheap enough for every turn.** The priming layer adds only a few hundred milliseconds per turn —
  roughly 70–400 ms in local benchmarks, depending on how much the walk surfaces — because it runs over a
  *local* embedding index against a warm daemon, with no network round-trip. Cheap enough to fire
  unconditionally rather than on demand.
- **Faithful pointers, not lossy summaries.** The workers that distill each past conversation into records
  follow one binding rule (the *dyad-anchor test*): keep only what a fresh AI couldn't already retrieve —
  your own reasoning, decisions, framings, and private facts, never public or textbook knowledge — as a
  compact record anchored back to its source, a *prior to verify* rather than a summary to trust.

It runs as a set of Claude Code **hooks** that prime every turn automatically — surfacing **semantic** and
**lexical** associations, seeded from *both* your message and the assistant's own previous reply. What
surfaces are compact **records** distilled from past conversations, sessions, and documents, embedded in a
vector space and reached by a multi-hop associative walk; new material is folded into the substrate offline,
during a *sleep* cycle.

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

## Configuration

Every tunable lives in [`config/settings.toml`](config/settings.toml), overlaid onto frozen defaults —
set a value to override it. The full inventory, by section:

**`[paths]`** — where state lives.

| Knob | Default | What it does |
|---|---|---|
| `storage_dir` | `"storage"` | Root for all runtime state — the canonical SQLite `graph.db`, the episodic log, the embeddings. |
| `exports_dir` | `"exports"` | Watch folder for claude.ai data-export archives (relative → under `storage_dir`); the sleep cycle scans it, then moves processed archives aside. |

**`[sleep]`** — the offline consolidation cycle.

| Knob | Default | What it does |
|---|---|---|
| `idle_minutes` | `30` | Idle-time threshold (minutes) gating an unattended sleep cycle. |
| `chunk_max_turns` | `120` | Max conversation turns folded into one episodic chunk. |
| `chunk_max_chars` | `30_000` | Per-chunk character budget — caps RAM when materializing chunks. |
| `mutex_timeout_s` | `300` | Lock timeout (seconds) guarding against two sleep cycles at once. |

**`[bridge]`** — the live spreading-activation walk that decides what gets primed each turn. These are the knobs you'll actually reach for.

| Knob | Default | What it does |
|---|---|---|
| `decay` | `0.8` | Multiplicative activation decay per hop — lower makes distant associations fade faster. |
| `min_score` | `0.3` | Minimum similarity for a link to be followed; prunes weak edges. |
| `k_per_query` | `10` | Nearest-neighbours fetched per seed. |
| `frontier_cap` | `10` | Max nodes carried on the frontier per hop. |
| `max_hops` | `4` | Maximum spreading depth from the seeds. |
| `bucket_total` | `25` | Total records primed per turn (semantic + lexical combined). |
| `bucket_lexical` | `5` | Cap on the lexical bucket (term / citation matches); the semantic budget is `bucket_total − bucket_lexical`. |
| `recency_strength` | `0.25` | Weight of the recency bias, `[0,1]` — `0` disables it, higher favours recent records. |
| `recency_age_span_days` | `180` | Age-normalisation span (days) for the recency penalty. |
| `recency_p_max` | `0.5` | Ceiling of the recency penalty. |
| `recency_filter_cutoff` | `""` | Hard date cutoff — records older than this are excluded outright; `""` = off. |
| `max_records` | `20` | Output cap for the deliberate `spread()` surface (MCP/CLI); equals the semantic budget. |

**`[vec_index]`** — the embedding transport (derived, rebuildable).

| Knob | Default | What it does |
|---|---|---|
| `model_name` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | The `fastembed` embedding model — multilingual MiniLM, 384-dim (covers EN + RO). |
| `persist_dir` | `"vec_index/chroma"` | ChromaDB persistent directory (relative → under `storage_dir`). |

MiniLM is the default for speed. For better retrieval geometry at ~+1 GB RAM, `BAAI/bge-m3`
is a drop-in upgrade — see [docs/embedding-options.md](docs/embedding-options.md) for the
quality-vs-RAM trade-off and the switch procedure (custom-model registration + dimension
rebuild).

**`[llm]`** — the offline distillation step.

| Knob | Default | What it does |
|---|---|---|
| `model` | `""` | Pins the model for distillation; empty uses the Claude Code default. |
| `auth_token_env` | `"CLAUDE_CODE_OAUTH_TOKEN"` | Env-var name the SDK reads for the OAuth token (only the unattended scheduler needs it). |

**`[mcp]`** — the read surface.

| Knob | Default | What it does |
|---|---|---|
| `read_only` | `true` | The MCP server exposes read-only access; every write goes through the CLI. Leave it on. |

## A note on framing

The neuroscience vocabulary here (System 1, spreading activation, priming, consolidation, sleep) is
used as **analogy and design lineage** (the architecture sits on the MINERVA-2 / multiple-trace and
spreading-activation literature), not as a claim of biological equivalence or that the AI possesses a
real System 1. `source_date` is when the *conversation* happened, not when a fact became true — treat
surfaced records as a lossy *pointer* to their source, a prior to verify, not evidence.

## Changelog

The format follows [Keep a Changelog](https://keepachangelog.com); the project uses
[Semantic Versioning](https://semver.org). All 0.x releases are proof-of-concept —
minor versions may still move fast.

### 0.2.0 — 2026-07-13

**Added**

- **Cross-turn priming dedup.** A record primed in the last *N* turns of a session
  (default 10, `bridge.dedup_window_turns`) is no longer re-injected; the freed budget
  backfills from the walk's tail, so more distal associations surface instead of repeats.
  The per-turn total stays same-or-fewer, never padded. Disable with
  `PRIMING_STREAM_DEDUP_OFF`.
- **BGE-M3 embedder option.** `BAAI/bge-m3` (1024-dim) is now documented as a drop-in,
  higher-quality alternative to the default MiniLM (384-dim) — better retrieval geometry
  at ~+1 GB RAM. See [docs/embedding-options.md](docs/embedding-options.md) for the
  quality-vs-RAM trade-off and the switch procedure.

**Changed**

- **Conditional verify footer.** The salient-context footer now asks the model to verify a
  cited record via `graph_chunk_around_anchor` *only when that tool is connected*, and to
  mark the specific as unverified otherwise — no fabricated verification when the read-only
  MCP server is absent.
- **Faster MCP startup.** `chromadb` is imported lazily, so the read-only MCP server (which
  never builds an index) no longer pays the import cost on startup.
- **Extraction contract.** Transferable-learning records now keep their origin entity
  verbatim in the body as a lexical recall seed — including internal-tool, device, and
  script origins, not just clients.

### 0.1.0 — 2026-06-30

- Initial public release.

## License

[MIT](LICENSE).
