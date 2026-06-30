"""End-to-end Priming Stream per-turn overhead vs baseline.

Priming Stream adds wall-time to each user turn via the UserPromptSubmit hook
(`python -m priming_stream.hooks.user_prompt_submit`): a fresh Python process that
hits the warm daemon, runs the spreading walk over the substrate,
renders the salient-context block, and emits it. This measures that, against
the Python-startup floor (the part a native-client hook would cut).

All against the LIVE warm daemon (read-only).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N = 12
WARMUP = 3

# Replace these with prompts that match YOUR substrate: a couple that should hit
# corpus-relevant records, one short filler, one clearly off-topic. "corpus-relevant"
# latency depends on what your substrate actually contains.
PROMPTS = {
    "corpus-relevant A":
        "how does spreading activation surface associatively related memories",
    "corpus-relevant B":
        "the bridge daemon architecture and the sleep-cycle write path",
    "short generic": "ok, let's continue",
    "off-corpus (cooking)": "how do I make a basic tomato sauce from scratch",
}


def _time_cmd(args, stdin_text=None, n=N, warmup=WARMUP):
    times = []
    for i in range(n + warmup):
        t0 = time.perf_counter()
        subprocess.run(
            args, cwd=str(ROOT),
            input=(stdin_text.encode("utf-8") if stdin_text is not None else None),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:
            times.append(dt)
    times.sort()
    return times


def _stats(ts):
    n = len(ts)
    mean = sum(ts) / n
    p50 = ts[n // 2]
    p90 = ts[int(n * 0.9)]
    return f"mean={mean:6.1f}ms  p50={p50:6.1f}  p90={p90:6.1f}  min={ts[0]:6.1f}  max={ts[-1]:6.1f}"


def main() -> None:
    py = sys.executable
    print(f"N={N} timed runs each (+{WARMUP} warmup discarded), live warm daemon\n")

    # 1. Python interpreter startup floor (paid regardless; native-hook target)
    base = _time_cmd([py, "-c", "pass"])
    print(f"[baseline] python -c pass    {_stats(base)}")
    base_mean = sum(base) / len(base)

    # 2. The full hook, per prompt (the real per-turn Priming Stream overhead)
    hook = [py, "-m", "priming_stream.hooks.user_prompt_submit"]
    print()
    for label, prompt in PROMPTS.items():
        stdin = json.dumps({"prompt": prompt, "session_id": "speedtest"})
        ts = _time_cmd(hook, stdin_text=stdin)
        mean = sum(ts) / len(ts)
        print(f"[hook] {label:36s} {_stats(ts)}")
        print(f"        -> Priming Stream bridge work over startup ~ {mean - base_mean:5.1f}ms")
    print()
    print("Priming Stream per-turn overhead = the full hook wall-time (there is no hook without Priming Stream).")
    print("The (hook - baseline) delta is the spreading-walk+render+daemon-roundtrip cost;")
    print("the baseline is the Python process startup a native-client hook would remove.")


if __name__ == "__main__":
    main()
