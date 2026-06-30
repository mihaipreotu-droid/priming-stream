"""``prime sample-export`` — pick N conversations from a Claude.ai export.

Selection is **size-stratified, deterministic** per spec §1.5: 5 large
(top by ``total_chars``), 10 medium (contiguous around the median), 5 small
(smallest with ``total_chars > 1000``). No randomness. Inputs that have
fewer than N usable conversations fall back to whatever is available and
print a warning instead of erroring.

Input: either a directory containing ``conversations.json`` (typical
shape of an unzipped claude.ai export) or the ``conversations.json`` file
itself.

Output: ``<out_dir>/conversations.json`` — a subset list of conversation
objects, otherwise byte-shaped like the input (downstream
``ClaudeAiExportAdapter`` already handles directory + bare-json inputs).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "sample-export",
        help="select N size-stratified conversations from a claude.ai export",
    )
    p.add_argument(
        "--n", type=int, default=20,
        help="number of conversations to keep (default 20)",
    )
    p.add_argument(
        "--in", dest="in_path", required=True,
        help="path to conversations.json or its containing directory",
    )
    p.add_argument(
        "--out", required=True,
        help="output directory; writes <out>/conversations.json",
    )
    p.set_defaults(func=_cmd_sample_export)


def _resolve_input(in_path: Path) -> Path:
    """Return the path of the actual ``conversations.json`` file."""
    if in_path.is_dir():
        target = in_path / "conversations.json"
        if not target.is_file():
            raise FileNotFoundError(
                f"conversations.json not found in directory {in_path}"
            )
        return target
    if in_path.is_file():
        return in_path
    raise FileNotFoundError(f"input path does not exist: {in_path}")


def _total_chars(conv: dict) -> int:
    msgs = conv.get("chat_messages") or []
    if not isinstance(msgs, list):
        return 0
    total = 0
    for msg in msgs:
        if not isinstance(msg, dict):
            continue
        text = msg.get("text", "")
        if isinstance(text, str):
            total += len(text)
    return total


def select_sample(conversations: list[dict], n: int) -> list[dict]:
    """Deterministic size-stratified pick.

    Layout (when N==20): 5 large + 10 median + 5 small. For other N, the
    strata scale proportionally (1/4, 1/2, 1/4) with rounding so the sum
    is exactly N. Small bucket filters ``total_chars > 1000`` to skip
    empties; if it can't fill, the deficit is dropped and a warning is
    surfaced by the caller.
    """
    # Annotate then sort by total_chars descending. Stable on ties.
    annotated = sorted(
        ((_total_chars(c), idx, c) for idx, c in enumerate(conversations)),
        key=lambda t: (-t[0], t[1]),
    )
    total = len(annotated)
    if total == 0 or n <= 0:
        return []
    if total <= n:
        # Not enough material to stratify — return everything in original order.
        return [c for _, idx, c in sorted(annotated, key=lambda t: t[1])]

    # Strata sizes: 25% large, 50% median, 25% small. Round so they sum to n.
    n_large = max(1, n // 4)
    n_small = max(1, n // 4)
    n_median = n - n_large - n_small
    if n_median < 0:
        n_median = 0

    large = annotated[:n_large]

    # Median band: contiguous slice centred on the median index.
    mid = total // 2
    half = n_median // 2
    start = max(0, mid - half)
    end = start + n_median
    if end > total:
        end = total
        start = max(0, end - n_median)
    median = annotated[start:end]

    # Small: smallest with total_chars > 1000, descending in size from the
    # bottom. Iterating from the tail gives us the smallest-first ordering.
    small_pool = [t for t in reversed(annotated) if t[0] > 1000]
    small = small_pool[:n_small]

    # Dedupe across strata by original index, preserving stratum priority
    # (large first, then median, then small). A conversation that lands in
    # the small bucket but is also in median (rare on small inputs) keeps
    # the earlier slot only.
    picked: dict[int, dict] = {}
    for _, idx, c in large:
        picked[idx] = c
    for _, idx, c in median:
        picked.setdefault(idx, c)
    for _, idx, c in small:
        picked.setdefault(idx, c)

    # Return in original input order — downstream adapter doesn't care, but
    # stable output is friendlier to diff.
    return [picked[idx] for idx in sorted(picked.keys())]


def _cmd_sample_export(args: argparse.Namespace) -> int:
    try:
        in_path = Path(args.in_path).expanduser()
        out_dir = Path(args.out).expanduser()
        target = _resolve_input(in_path)
        data = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            print(
                f"[sample-export] conversations.json must be a list; got "
                f"{type(data).__name__}",
                file=sys.stderr,
            )
            return 1

        picked = select_sample(data, int(args.n))
        if len(picked) < int(args.n):
            print(
                f"[sample-export] warn: requested {args.n} but input "
                f"yielded only {len(picked)} eligible conversations",
                file=sys.stderr,
            )

        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "conversations.json"
        out_file.write_text(
            json.dumps(picked, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(
            f"[sample-export] wrote {len(picked)} conversations -> {out_file}"
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"sample-export failed: {exc}", file=sys.stderr)
        return 1
