# Embedding options — MiniLM (default) vs BGE-M3 (quality upgrade)

Priming Stream embeds every record's summary locally (`fastembed` + ChromaDB, no network
round-trip) and runs the spreading-activation walk over that embedding space. The embedder
is therefore the single component that decides *where records sit relative to each other* —
and thus which records the walk can reach within the priming budget.

Two models are documented here. **MiniLM is the shipped default**; **BGE-M3 is a drop-in
quality upgrade** that trades roughly **+1 GB of resident RAM and ~+50 ms/query** for
materially better retrieval geometry. The choice is a straight **quality vs. RAM/latency**
trade — both run on plain CPU via `fastembed`, with no GPU dependency.

| | **MiniLM** (default) | **BGE-M3** (upgrade) |
|---|---|---|
| Model id | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | `BAAI/bge-m3` |
| Dimensions | 384 | 1024 |
| Pooling / config | mean pooling | CLS pooling + normalize, no query/passage prefixes |
| Latency / query (CPU) | ~5 ms | ~57 ms |
| Resident RAM (one model in the daemon) | ~0.55 GB | ~1.5 GB |
| Index size (Chroma) | 1× | ~2.7× (1024 vs 384 dim) |
| License | Apache 2.0 | Apache 2.0 |

Both are multilingual and cover short EN + RO summaries, which is what this substrate stores.

## Why you might upgrade — the quality axis

MiniLM was chosen at the vector-index stage for **speed**, not retrieval quality. The bridge
either wins or loses on one thing: whether the spreading walk surfaces the *bridge record*
(the associatively-relevant record that ties the current prompt to something learned earlier)
**inside the priming budget**. When the target sits just below the budget cutoff, priming
misses it — and that is a property of the **embedding geometry**, i.e. which model placed the
records in the space, not of the walk's tuning (the walk knobs plateau).

On a rank-probe over the hard cases (position of the bridge record; lower = better), BGE-M3:

- **recovers targets MiniLM buries** — cases where the bridge record ranked in the 30s–40s
  under MiniLM move into the top ~1–10 under BGE-M3;
- shows **no regressions** on the cases MiniLM already got right;
- gives the **widest separation** between related and unrelated pairs (a clean canary margin),
  which is what makes the activation threshold behave predictably.

So BGE-M3's value is concentrated exactly where MiniLM is weakest: the buried-bridge case that
passive priming exists to catch. If your substrate is small or your prompts rarely depend on
distal associations, MiniLM's geometry is already fine — and 10× cheaper per query.

## The cost axis — RAM and latency

- **RAM: +~1 GB resident.** One model is loaded in the daemon regardless of how many sessions
  are active, so this is a fixed, one-time cost of ~0.55 GB → ~1.5 GB. On a 16 GB machine it is
  negligible; on a memory-constrained host it is the main reason to stay on MiniLM.
- **Latency: +~50 ms/query, ~+100 ms/turn on the hook hot path.** The per-query embed grows
  from ~5 ms to ~57 ms; end to end the priming hook stays well within a sub-second budget. The
  first prompt after a cold start also pays a one-off model load (~1.5–3 s), same pattern as
  MiniLM, covered by the daemon autostart + FTS5 fallback.
- **Storage:** 1024-dim vectors make the Chroma collection ~2.7× larger — negligible at the
  scale of a personal substrate.

## Switching to BGE-M3

BGE-M3 is **not** in `fastembed`'s stock model list, so it must be registered as a custom
model before use, and the switch is a **dimension change** (384 → 1024), which means the
Chroma collection has to be **rebuilt, not upserted**.

1. **Register the custom model** (at import time, e.g. in `integrations/vec_index.py` or a
   config-driven registry) so the daemon + CLI can select `BAAI/bge-m3`:

   ```python
   from fastembed import TextEmbedding
   from fastembed.common.model_description import PoolingType, ModelSource

   TextEmbedding.add_custom_model(
       model="BAAI/bge-m3",
       pooling=PoolingType.CLS,
       normalization=True,
       sources=ModelSource(hf="BAAI/bge-m3"),
       dim=1024,
       model_file="onnx/model.onnx",
       additional_files=["onnx/model.onnx_data"],  # ⚠️ LOAD-BEARING — see below
   )
   ```

   **⚠️ `additional_files=["onnx/model.onnx_data"]` is mandatory.** BGE-M3 ships in ONNX
   *external-data* format: `model.onnx` (~0.7 MB, graph skeleton only) plus `model.onnx_data`
   (~2.2 GB, the actual weights). Omit the sidecar and `fastembed` fetches only the skeleton →
   a **weightless model that loads without error and emits garbage embeddings** (silent
   corruption, no exception).

2. **Integrity gate — run before any re-embed.** Because a weightless load is silent, verify
   the model is complete *before* rebuilding the live index (a garbage re-embed corrupts
   retrieval with no error):
   - **On disk:** `model.onnx_data` present and ~2.2 GB (not just the 0.7 MB `model.onnx`).
   - **Semantic canary:** embed one related pair and one unrelated pair; require the related
     cosine to exceed the unrelated one by a clear margin (a weightless model gives ~random
     cosines). Proceed only if both checks pass.

3. **Point the config at it:** in `config/settings.toml`, set
   `[vec_index] model_name = "BAAI/bge-m3"`.

4. **Rebuild the collection (drop + recreate, not upsert).** A Chroma collection is created at
   a fixed dimension; 1024-dim vectors cannot upsert into a 384-dim collection (it errors).
   Delete the existing `storage/vec_index/chroma` (or ensure your rebuild recreates rather than
   reuses the collection), then run `prime vec-index-rebuild`. Take a `prime db-snapshot`
   first — the canonical text lives in SQLite and is untouched by the swap.

5. **Restart the daemon** (`prime daemon restart`) so it loads the new model (first load
   ~1.5–3 s, then cached), and spot-check that a few known association cases now land inside
   the priming budget.

## Reversibility

Fully reversible. SQLite (`graph.db`, the canonical record text) is never touched by an
embedder swap — only the derived Chroma index changes. To revert, set `model_name` back to the
MiniLM id and run `prime vec-index-rebuild`.

## Note on the reconcile cutoff

BGE-M3's background cosine distribution sits higher than MiniLM's, which compresses the gap
between near-duplicate records and unrelated ones. The sleep-cycle reconcile step gates
candidate pairs on a cosine cutoff; under BGE-M3 a higher background mainly means **more
candidate pairs reach the LLM judge at sleep time** (a cost, bounded by the top-k cap), not a
correctness problem. If you switch and rely heavily on automatic supersedence/dedup, re-check
the reconcile cutoff against your own near-duplicate vs. unrelated cosines rather than assuming
the MiniLM-era value transfers.
