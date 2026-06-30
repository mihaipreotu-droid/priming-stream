# Architecture

Priming Stream is a persistent, **owner-controlled associative memory** for an AI agent working in
Claude Code. This document explains how it works. For install and usage, see [SETUP.md](SETUP.md);
for the honest framing of the neuroscience vocabulary, see the end of this file.

## Two memory systems

- **Episodic** — the raw record of what happened. Conversation transcripts and ingested documents are
  split into chunks and kept **immutable** on disk (`storage/corpus/imports/…/<chunk_id>.md`). Nothing
  rewrites them; they are the evidence layer.
- **Semantic** — the durable, associative layer. Compact **memory records** distilled from the episodic
  layer: short summaries (≤~20 words) plus a pointer back to the chunk + character offset they came
  from. This is the substrate the bridge primes from.

Records come in two kinds:

- `claim` — a notable moment distilled from a conversation (a decision, an insight, a conceptual
  distinction, a finding).
- `index_card` — a content-only summary of a document, keyed by a canonical `doc_key`.

There are **no nodes, edges, keywords, or hand-built graph**. A record carries its meaning in its
summary embedding; associations emerge from proximity in embedding space. "Plasticity" is just corpus
growth: more records on a topic ⇒ a denser neighborhood ⇒ stronger retrieval (a multiple-trace /
MINERVA-2 lineage, not a typed knowledge graph).

## Storage model

`storage/graph.db` (SQLite) is the **canonical** source of truth:

```
storage/
  graph.db          ← canonical: records (+ records_staging, records_trash) + FTS5 index + sleep_cycles audit
  corpus/
    imports/<source>/<session>/<chunk_id>.md   ← episodic chunks (immutable evidence)
    _cursor.json                                ← materialize cursor
  episodic/
    chunks.jsonl, live_events.jsonl, …          ← episodic log + hook telemetry
  vec_index/chroma/                             ← ChromaDB vector index (derived, rebuildable from graph.db)
```

- **`records`** holds the live substrate. **`records_staging`** holds a cycle's incoming records before
  they are promoted (invisible to priming until then). **`records_trash`** holds soft-deleted records
  (reversible via `prime record restore`).
- **ChromaDB** (local embeddings via `fastembed`, MiniLM-L12-v2, 384-dim) and the **SQLite FTS5** index
  are both **derived** from `graph.db` and rebuildable (`prime vec-index-rebuild`). The embeddings run
  in-process — no third-party API, no remote calls.
- Backups are local snapshots of the DB (`prime db-snapshot`); the substrate never leaves the machine.

## The bridge — read-time priming

On each prompt, a lightweight `UserPromptSubmit` hook asks a resident local **daemon** (which keeps the
embedding model and index warm across sessions) to run a **spreading-activation walk** over the
embedding space and inject the most associatively relevant records as a "Salient context" block. This is
**automatic and unconditional** — priming is pushed as background context on every turn, not fetched
on demand when the agent decides to query memory. (Claude Desktop, which has no hook, instead *pulls*
the same bridge via an MCP tool — see SETUP.md §5.)

- **Two seeds.** The walk runs from the user prompt and from the previous assistant turn separately,
  then combines them — so a short pivot prompt isn't drowned by a long prior answer.
- **Multi-hop spread.** From each activated record it propagates to records similar to *it*, with
  multiplicative decay per hop; the walk stops when activation falls below a threshold or `MAX_HOPS` is
  reached. (Mathematically a random-walk-with-restart / personalized-PageRank over implicit similarity
  edges.)
- **Two output buckets.** *Bucket A* is the semantic spread (associative priming). *Bucket B* is a
  lexical FTS5 match over the prompt only (so naming a specific paper or rare term surfaces it even when
  dense embeddings would bury it). B is deduped against A and capped by budget, not by a relevance
  threshold — filtering happens at output, in the live reasoning, not at intake.
- **Recency.** Each record carries a `source_date` (when the conversation happened). It is surfaced in
  the output, and a gentle recency weight breaks near-ties in favor of newer records — never as a hard
  filter.

Tunable knobs (`config/settings.toml`): `DECAY`, `MIN_SCORE`, `FRONTIER_CAP`, `K_PER_QUERY`,
`MAX_HOPS`, `MAX_RECORDS`. Warm hook latency is on the order of a few hundred milliseconds.

## The write path — the sleep cycle

The durable layer is written **only** offline, in a batched "sleep cycle" — never per turn. One skill,
`/prime-ingest`, runs the whole cycle:

1. **Ingest** a source (Claude Code session transcripts, claude.ai export archives, or documents) into
   the episodic log, or drain whatever is already pending.
2. **Extract** — one worker per conversation reads it whole and distills records against a binding
   extraction contract (be selective, distill the conversation's *output* not the transcript, skip
   anything retrievable from general knowledge). Workers run on a smaller model by default, with large
   conversations routed to a larger one.
3. **Reconcile** — incoming records are checked against the substrate by a conservative judge:
   exact-duplicate documents auto-merge; near-clones collapse; a direct contradiction supersedes the
   false/older record. The bar for deletion is high; doubt ⇒ keep both.
4. **Finalize** — staged records are promoted into `records`, the vector index updates, and the daemon
   reloads.

A scheduled task can run the cycle unattended on idle. The judgment steps (extraction, reconciliation)
are run as Claude Code **Workflow** agents against your Claude subscription — there is no separate API
key and no local generative model.

## Document ingestion

`/prime-ingest --ingest <path>` ingests documents (a file, a folder, or scattered files) as `index_card`
records. Non-Markdown files are converted with `markitdown`; native `.md`/`.txt` are read in place. A
canonical `doc_key` (`doi:` > `url:` > a title-derived key) gives each document one identity, so a
conversational stub and a later file card for the same document reconcile into one node. Documents a
conversation builds on are captured automatically as cards in the same cycle.

## Curation and inspectability

- **Curation is owner-driven and CLI-only.** `prime record create | edit | delete | restore` lets the
  owner author, correct, or remove records by hand, live, with immediate re-embedding. The automatic
  flow only ever writes to the episodic layer.
- **The MCP server is strictly read-only** — every write goes through the CLI, never through MCP.
- **Inspect** the substrate with `prime dashboard` (browser + corpus health), `prime echoes` (what
  was primed, per turn), and `prime search` (lexical / semantic recall).

The data flows one way for everything automatic: conversation → episodic (immutable) → sleep cycle →
records (mutable) → bridge → live priming. Curation is the only deliberate, owner-invoked write over the
substrate.

## A note on framing

The neuroscience vocabulary here — *System 1*, spreading activation, priming, consolidation, sleep — is
used as **analogy and design lineage**, drawn from the multiple-trace (MINERVA-2) and spreading-
activation literature. It is **not** a claim that the AI has a real System 1, a subconscious, or any
biological equivalent. Read the system as an **external, System-1-*like* associative memory** that the
owner controls and the agent draws on — a tool, not a faculty the model possesses.

Two honesty notes on the lineage: the **spreading-activation** retrieval is uncommon in agent memory
generally (most systems use plain vector or keyword lookup) but shared with a few contemporary
cognitive-memory systems — a real capability, not an invention of this project; and Priming Stream does the
*fast-retrieval* half of memory (surfacing relevant stored traces) — **not** consolidation into the
model's weights. It is a retrieval-and-hygiene layer, not "expertise development."

Treat a surfaced record as a **lossy pointer** to its source and a **prior to verify**, not as evidence:
its summary is a ≤20-word distillation, and its `source_date` is when the *conversation* happened, not
when a fact became true.

## Scope

Proof-of-concept: **single-user, owner-controlled, Windows-first, Claude-bound.** Developed and tested
on Windows — the code is cross-platform Python (only the unattended scheduler is Windows-specific, with a
cron alternative), but other OSes are untested. No multi-user, cloud sync, mobile, or real-time graph
updates — every durable change goes through the sleep cycle by design.
