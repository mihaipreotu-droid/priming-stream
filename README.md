# Priming Stream

> *Associative memory for AI agents.*

[![CI](https://github.com/mihaipreotu-droid/priming-stream/actions/workflows/ci.yml/badge.svg)](https://github.com/mihaipreotu-droid/priming-stream/actions/workflows/ci.yml)

Most memory systems for AI agents solve **known-unknowns**: you know what you don't know, so you (or
the model) formulate a query and search for it — RAG, memory-search tools, retrieval on demand.
Priming Stream solves **unknown-unknowns**: the things you don't know you don't know, where no query
will ever be issued because nothing signals there is something to look for. Instead of waiting to be
asked, it **pushes** — unconditionally, on every turn — whatever in the accumulated memory is
associatively related to what is happening right now: sometimes close to the matter at hand, sometimes
far from it, surfaced precisely because nobody thought to ask.

The mechanism is borrowed from human memory, which is associative and spontaneous: as you perceive or
reason, related memories — episodic and semantic — surface on their own. **Priming Stream** gives an AI
agent working in [Claude Code](https://docs.claude.com/claude-code) something of that capacity.

Two properties make it practical to run for real:

- **Cheap enough for every turn.** Through a warm local daemon, short prompts prime in ~100–250 ms
  wall and even a 3,200-char prompt returns in ~840 ms, under a hard 2-second client deadline with a
  lexical (BM25) fallback beyond it — because it runs over a *local* embedding index, with no network
  round-trip. Cheap enough to fire unconditionally rather than on demand.
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
| `dedup_window_turns` | `10` | Cross-turn dedup: records primed in the last *N* turns of a session are suppressed so freed slots surface fresh ones; `0` = off. |
| `seed_char_budget` | `5000` | User-first input cap for the embedding seed: the user prompt is **never** truncated; the response-seed takes what remains. Sized so the embed stays under the 2 s client deadline; `0` = no cap. |
| `turn_floor` | `0.40` | Turn-gate master switch: a turn whose top rank-score falls under the floor *whispers* (see below); `0` switches the **entire** gate off — the one-knob rollback. |
| `regime_density` | `0.6` | Tool-density threshold above which a turn counts as execution regime and whispers. |
| `whisper_k` | `5` | Semantic records kept on a whispered turn. |
| `whisper_lex_k` | `3` | Lexical records kept on a whispered turn. |
| `kickoff_turns` | `3` | The first *N* turns of a session always prime full, unconditionally. |

Two run-time behaviours worth knowing around this table. **The turn-gate** (`turn_floor` > 0): one
mechanism, two outputs — every turn primes either **full** or as a **whisper** (top `whisper_k` +
`whisper_lex_k` records), never silence. A turn whispers on any of three triggers: weak associative
field (top score under `turn_floor` — rendered with an explicit "weak associative field" marker so the
model treats those records as weak suggestions), execution regime (`tool_density ≥ regime_density`),
or a `<task-notification>` turn. It applies only to the automatic hook path — deliberate MCP/CLI pulls
are never gated. **The response-seed**: the walk is seeded from *both* your message and the tail
(~1,200 chars) of the assistant's previous reply, recovered from the session transcript; the seed
budget is allocated user-first, and the hook's daemon round-trip runs under a fixed 2,000 ms deadline
(`daemon/client.py`), falling back to lexical search past it. `prime echoes --stats` shows the
per-day health of all of this: empty-rate, latency percentiles (uncensored, breaches included),
whisper counts, and the response-seed's health.

**`[vec_index]`** — the embedding transport (derived, rebuildable).

| Knob | Default | What it does |
|---|---|---|
| `model_name` | `onnx-community/bge-m3-ONNX-int8` | The `fastembed` embedding model — int8-quantized BGE-M3, 1024-dim, multilingual, custom-registered automatically. |
| `persist_dir` | `"vec_index/chroma"` | ChromaDB persistent directory (relative → under `storage_dir`). |

The shipped embedder is **int8-quantized BGE-M3** — the model the system actually runs on and is
measured against. BGE-M3's geometry is what makes distal association work (it recovers bridge records
that smaller models bury); the int8 quantization is what makes it affordable on the hot path. Measured
on CPU, in-process, median of 7:

| prompt chars | 40 | 400 | 800 | 1,600 | 3,200 |
|---|---|---|---|---|---|
| embed latency | 21 ms | 81 ms | 153 ms | 360 ms | 912 ms |

Through the live daemon a 3,200-char prompt returns in ~840 ms wall — under the 2 s deadline that
used to be breached by fp32 (~2.1 s embed alone). Fixed costs vs fp32: **~0.98 GB** resident RAM
(vs ~1.55 GB), **1.8 s** load+warmup (vs 5.1 s), **568 MB** self-contained on disk (vs ~2.27 GB),
full re-embed of a ~6.5k-record collection in **~13 min** (vs ~54 min). Rank parity against fp32:
one boundary flip out of 155 association targets — quantization jitter, no systematic loss. The
daemon guards embedder identity with a **semantic canary at every warmup** (the HF revision cannot
be pinned by `fastembed`; a wrong artifact refuses to serve rather than emit garbage). Details,
procedures, and the fp32 rollback: [docs/embedding-options.md](docs/embedding-options.md).

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

### 0.3.0 — 2026-07-22

**Added**

- **int8 BGE-M3 embedder as the shipped default.** `onnx-community/bge-m3-ONNX-int8`
  (1024-dim, custom-registered automatically): BGE-M3 retrieval geometry at roughly half
  the RAM and 2.3–3× the speed of fp32 — a 3,200-char prompt embeds in ~912 ms instead of
  ~2.1 s. A **semantic canary at every daemon warmup** (startup + reload) guards embedder
  identity: on failure `/v1/spread` refuses and the hook degrades to lexical fallback
  instead of serving garbage similarities. Measured numbers and procedures in
  [docs/embedding-options.md](docs/embedding-options.md).
- **Response-seed.** The walk's second seed is now live: the hook recovers the assistant's
  previous reply from the session transcript tail and feeds its last ~1,200 chars to the
  semantic seed (never the lexical bucket). The slice shape was chosen empirically —
  tail-only beat head+tail variants on carrier-injection coverage.
- **Turn-gate: full or whisper, never silence.** Every hook turn primes either full or as a
  whisper (top `whisper_k`=5 semantic + `whisper_lex_k`=3 lexical) on three triggers: weak
  associative field (top score < `turn_floor`, rendered with an explicit weak-field
  marker), execution regime (`tool_density ≥ regime_density`), or `<task-notification>`
  turns. First `kickoff_turns`=3 turns are exempt; unknown features fail open toward more
  priming; MCP/CLI pulls are never gated; `turn_floor = 0` rolls the whole gate back.
- **Uncensored priming telemetry.** Every echo line now carries `client_ms` (hook-side wall
  time — survives deadline breaches that censor `spread_ms`), `prompt_len`, `prev_len`,
  `seed_len`, and `gated`; `prime echoes --stats` renders per-day health: turns by source,
  empty-rate, latency percentiles with breach counts, whisper counts, response-seed health.
- **Daemon operability.** `prime daemon status --all` inventories every live daemon process
  and flags strays (running, listening, holding RAM, owning no endpoint — invisible to
  clients); hook-triggered autostart is rate-limited by a 90 s cooldown so a misfiring
  staleness check can't spawn a fleet (explicit `prime daemon start` bypasses it); daemon
  log lines carry `pid=` for per-instance attribution.
- **Extraction contract v4.1 — post-draft gates.** Extraction now runs three mechanical
  gates over the full draft list before emitting: compression (scripted word count, hard
  fail over 20), working-session residue (drop bookkeeping that leads with no transferable
  principle), and anchor verification (batched snippet-located offsets, never estimated).

**Changed**

- **Client deadline 800 ms → 2,000 ms, paired with a user-first seed budget.** The deadline
  is a ceiling on the tail, not a per-turn tax: typical turns are governed by
  `seed_char_budget` (5,000 chars; the user prompt is never truncated — the response-seed
  takes what remains, and only notification seeds may be cut).
- **Lexical fallback joins with OR + BM25.** The previous implicit-AND join required every
  prompt token to appear in a ≤20-word summary — structurally dead on prompts over ~15
  tokens, exactly the long-prompt breach case it exists for. Tokens are now deduped,
  capped at 64, OR-joined, BM25-ranked; a breach degrades to lexical priming instead of
  silence.
- **`/v1/reload` now swaps the full config.** `[bridge]` knob edits and the model name
  apply on the nightly reload, not only on a full restart — knob tuning and rollbacks
  behave as documented.
- **The write-cycle lock is a real OS lock** (`msvcrt`/`flock`), replacing an
  exists-then-write lockfile with a TOCTOU window; a crashed cycle releases implicitly.

**Fixed**

- An empty semantic bucket now whispers with the weak-field marker instead of slipping
  through as full, and the lexical fallback keeps the whisper cap on whispered turns
  instead of silently widening to 10 unmarked records.
- Unknown `turn_idx` (unreadable echo history) counts as kickoff — fail open toward full
  priming, never toward a lost exemption.
- Transcript tail reads tolerate a UTF-8 BOM; echoes state writes are atomic.
- The `VecIndexConfig` default now matches the live collection (a missing `settings.toml`
  could silently query a 1024-dim collection with a 384-dim model).
- A test fixture's hardcoded date aged out of its retention window and turned the suite
  permanently red; dates are now relative.

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
