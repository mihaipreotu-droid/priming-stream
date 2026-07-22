# Embedding model — int8-quantized BGE-M3 (what ships, and why)

Priming Stream embeds every record's summary locally (`fastembed` + ChromaDB, no network
round-trip) and runs the spreading-activation walk over that embedding space. The embedder
is the single component that decides *where records sit relative to each other* — and thus
which records the walk can reach within the priming budget.

The shipped default is **`onnx-community/bge-m3-ONNX-int8`** — a dynamic int8 quantization
of BAAI's BGE-M3 (1024-dim, CLS pooling + normalize, no query/passage prefixes), running on
plain CPU via `fastembed`, no GPU dependency. This is the model the system is developed and
measured against; everything below is measured, not estimated.

## Why BGE-M3 geometry

The bridge wins or loses on one thing: whether the walk surfaces the *bridge record* (the
associatively relevant record tying the current prompt to something learned earlier)
**inside the priming budget**. That is a property of the embedding geometry. On rank probes
over hard association cases, BGE-M3 recovers targets that smaller/faster models bury
(bridge records ranked in the 30s–40s move into the top ~1–10), with no regressions on the
easy cases, and gives a wide related-vs-unrelated separation — which is what makes the
activation threshold behave predictably.

## Why the int8 quantization

fp32 BGE-M3 on CPU is too slow for the hook's hot path on long prompts. The int8 export
keeps the geometry and cuts every operational cost roughly in half or better:

**Query latency, in-process (median of 7, CPU):**

| prompt chars | fp32 | **int8** | speedup |
|---|---|---|---|
| 40 | 48 ms | **21 ms** | 2.3× |
| 400 | 243 ms | **81 ms** | 3.0× |
| 800 | 460 ms | **153 ms** | 3.0× |
| 1,600 | 985 ms | **360 ms** | 2.7× |
| 3,200 | 2,143 ms | **912 ms** | 2.35× |

Through the live daemon (walk + HTTP overhead included), a 3,200-char prompt returns in
**~840 ms wall** — comfortably under the hook's 2 s client deadline; under fp32 the same
prompt breached the deadline and the turn got no semantic priming at all.

**Fixed costs:**

| | fp32 | **int8** |
|---|---|---|
| Resident RAM (daemon) | ~1.55 GB | **~0.98 GB** |
| Load + warmup (warm cache) | 5.1 s | **1.8 s** |
| On disk | 0.7 MB skeleton + 2.27 GB sidecar | **568 MB, self-contained** |
| Full re-embed of a ~6.5k-record collection | ~54 min | **~13 min** |

**Quality parity:** on a rank-parity probe against fp32 (155 in-budget association targets,
full-collection ranking), int8 kept top-10 overlap of ~8/10 with symmetric jitter of ±1–5
ranks and **one** boundary flip out of 155 — quantization noise, no systematic degradation.
The semantic canary separates cleanly (related ~0.80 vs unrelated ~0.40 on the English
probe pair).

## Operational notes (load-bearing)

- **Custom registration.** Neither BGE-M3 variant is in `fastembed`'s stock list; the
  registration lives in `integrations/vec_index.py` and runs automatically for the
  configured model. The int8 file is **self-contained** — no external-data sidecar.
- **The weightless trap (fp32 only).** fp32 BGE-M3 ships in ONNX *external-data* format:
  `model.onnx` (~0.7 MB skeleton) + `model.onnx_data` (~2.27 GB weights).
  `additional_files=["onnx/model.onnx_data"]` is **mandatory** there — omit it and
  `fastembed` fetches only the skeleton: a model that loads without error and emits garbage
  embeddings. The shipped registration handles both variants correctly.
- **Revision pinning + canary.** `fastembed` cannot pin an HF revision, so a wiped hub
  cache re-fetches whatever `main` points at. The daemon therefore runs a **semantic canary
  at every warmup** (startup *and* reload): a related pair + an unrelated control must
  separate cleanly, or `/v1/spread` refuses (503) and the hook degrades to its lexical
  fallback rather than serving garbage similarities. The development install pins HF
  revision `25b9af8e` in its local cache.
- **Dimension changes rebuild, not upsert.** A Chroma collection is created at a fixed
  dimension; switching models with a different dim means drop + recreate via
  `prime vec-index-rebuild`. Take a `prime db-snapshot` first — the canonical text lives in
  SQLite and is never touched by an embedder swap.

## Rollback

Fully reversible. Set `[vec_index] model_name = "BAAI/bge-m3"` (fp32) in
`config/settings.toml`, run `prime vec-index-rebuild` (~54 min on a ~6.5k-record
collection), and `prime daemon restart`. The registration for the fp32 variant is already
in the codebase as the rollback path.

## Note on the reconcile cutoff

BGE-M3's background cosine distribution sits higher than smaller models', which compresses
the gap between near-duplicate records and unrelated ones. The sleep-cycle reconcile step
gates candidate pairs on a cosine cutoff; a higher background mainly means **more candidate
pairs reach the LLM judge at sleep time** (a bounded cost), not a correctness problem. If
you rely heavily on automatic supersedence/dedup, check the cutoff against your own
near-duplicate vs. unrelated cosines.
