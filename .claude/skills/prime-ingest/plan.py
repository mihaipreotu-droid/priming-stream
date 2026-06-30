"""Sleep-cycle planner — turns a sleep-prepare manifest into per-conversation
extraction assignments for the conversational workflow.

Final design (converged 2026-05-31; routing revised 2026-06-10):
  - ONE worker per conversation (no segmentation, no framework-pass, no merge).
  - Route by CONTEXT LOAD, not chunk count: estimate body tokens per
    conversation; > TOKEN_THRESHOLD -> Opus single-pass. Default threshold
    is 0 — EVERY conversation goes to Opus (L(b) probe beta, 2026-06-10:
    with the same contract + chunking, Sonnet under-executes the
    analytical-moves class — corrections, decisions-with-rationale — that
    the substrate exists to keep; pass --threshold 100000 to restore the
    old size-based Sonnet routing).
  - Per-conversation assignment FILES (one small file per conv) so each worker
    reads only its own slice — non-chunk context overhead stays ~constant
    regardless of corpus size (refinement B).
  - Pre-generated unique rec_id pools (workers don't shell for ids).

Usage:  python plan.py <manifest.json> [--threshold 100000]
Writes: storage/corpus/_sleep_assign/<conv>.json  (per-conversation slices)
        storage/corpus/_sleep_index.json           (cycle_id + conv list + mode)
Prints: cycle_id + per-conversation line (mode, chunks, ~tokens).
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

from priming_stream.core.config import load_config
from priming_stream.core.models import new_record_id, now_iso
from priming_stream.core.paths import resolve_paths

_PN = re.compile(r"_p(\d+)$")
_UUID_SEG = re.compile(r"[/\\]([0-9a-fA-F-]{36}|u\d+)[/\\][^/\\]+$")

CHARS_PER_TOK = 3.8          # mixed RO/EN rough
# 100_000 -> conversations >100K body-tokens route to Opus, the rest to Sonnet.
# Reverted from 0 (all-Opus, L(b) 2026-06-10) on 2026-06-11: Opus showed weaker
# contract compliance (e.g. EN records on RO chunks, violating the language rule)
# + higher cost, while the manual quality verify found Sonnet records more
# grounded/findable; the analytical-moves edge didn't justify the swap.
# --threshold 0 forces all-Opus again. NOTE: routing to 'opus' does NOT grant a
# 1M window — the workflow alias 'opus' resolves to ~200K (CC bug #45169 strips
# the [1m] suffix on subagents); large convs rely on the worker's running-synthesis.
TOKEN_THRESHOLD = 100_000
SONNET_POOL = 50
OPUS_POOL = 150


def _conv_of(path: str) -> str:
    m = _UUID_SEG.search(path)
    return m.group(1) if m else Path(path).parent.name


def _order_key(chunk_id: str) -> int:
    m = _PN.search(chunk_id)
    return int(m.group(1)) if m else -1


def _body_len(path: str) -> int:
    try:
        txt = Path(path).read_text(encoding="utf-8")
    except OSError:
        return 0
    if txt.startswith("---"):
        end = txt.find("\n---", 3)
        if end != -1:
            nl = txt.find("\n", end + 1)
            return len(txt[nl + 1:]) if nl != -1 else 0
    return len(txt)


def main() -> None:
    args = sys.argv[1:]
    manifest_path = args[0]
    threshold = TOKEN_THRESHOLD
    if "--threshold" in args:
        threshold = int(args[args.index("--threshold") + 1])

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8-sig"))
    cycle_id = manifest["cycle_id"]
    prepared = manifest.get("prepared_chunks", [])

    cfg = load_config()
    paths = resolve_paths(cfg)
    corpus = Path(paths.graph_db).parent / "corpus"
    contract_path = str(Path(__file__).resolve().parents[3] / "prompts" / "extract_record.md")
    assign_dir = corpus / "_sleep_assign"
    if assign_dir.exists():
        shutil.rmtree(assign_dir)
    assign_dir.mkdir(parents=True, exist_ok=True)

    results_dir = corpus / "_sleep_results"   # workers each write ONE <conv>.json here
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    by_conv: dict[str, list[dict]] = {}
    for c in prepared:
        by_conv.setdefault(_conv_of(c["path"]), []).append(c)

    seen: set[str] = set()
    def gen_ids(n: int) -> list[str]:
        out = []
        while len(out) < n:
            rid = new_record_id()
            if rid not in seen:
                seen.add(rid); out.append(rid)
        return out

    created_at = now_iso()
    index = []
    for conv, chunks in sorted(by_conv.items(), key=lambda kv: -len(kv[1])):
        chunks.sort(key=lambda c: _order_key(c["chunk_id"]))
        body_chars = sum(_body_len(c["path"]) for c in chunks)
        est_tokens = int(body_chars / CHARS_PER_TOK)
        mode = "opus" if est_tokens > threshold else "sonnet"
        slice_obj = {
            "conv": conv,
            "mode": mode,
            "est_tokens": est_tokens,
            "results_dir": str(results_dir),
            "contract_path": contract_path,
            "created_at": created_at,
            "chunks": [
                {"chunk_id": c["chunk_id"], "path": c["path"], "source_uri": c["source_uri"]}
                for c in chunks
            ],
            "rec_ids": gen_ids(OPUS_POOL if mode == "opus" else SONNET_POOL),
        }
        assign_path = str(assign_dir / f"{conv}.json")
        Path(assign_path).write_text(json.dumps(slice_obj, ensure_ascii=False), encoding="utf-8")
        index.append({"conv": conv, "mode": mode, "assign_path": assign_path,
                      "est_tokens": est_tokens, "n_chunks": len(chunks)})

    index_path = corpus / "_sleep_index.json"
    index_path.write_text(json.dumps({"cycle_id": cycle_id, "conversations": index},
                                     ensure_ascii=False), encoding="utf-8")

    print(f"cycle_id={cycle_id}")
    print(f"index={index_path}")
    print(f"conversations={len(index)} chunks={len(prepared)} threshold={threshold} tok")
    for e in index:
        print(f"  {e['conv'][:13]:13s} {e['mode']:6s} chunks={e['n_chunks']:2d} ~tok={e['est_tokens']}")


if __name__ == "__main__":
    main()
