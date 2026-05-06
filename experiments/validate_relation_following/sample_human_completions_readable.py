"""
Randomly sample human relational completions and print plain English analogies to stdout:

  <A> is to <B> as <C> is to <D>

where D is the participant's answer.

  python sample_human_completions_readable.py
  python sample_human_completions_readable.py --n 50 --seed 42 > sample.txt
  python sample_human_completions_readable.py --out my_sample.txt   # also save a copy
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "human_completions_flat.json"


def _load_completions(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    comps = data.get("completions")
    if not isinstance(comps, list):
        raise ValueError(f"No completions list in {path}")
    return comps


def _format_block(i: int, n: int, c: Dict[str, Any]) -> str:
    a, b, cc, d = (str(c.get(k, "") or "").strip() for k in ("A", "B", "C", "D"))
    rel_name = str(c.get("relation_name", "") or "").strip()
    rel_key = str(c.get("relation_key", "") or "").strip()
    tid = c.get("triple_id", "")
    cid = str(c.get("completion_id", "") or "").strip()
    elig = c.get("participant_eligible_complete")
    conf = str(c.get("confidence", "") or "").strip() or "—"

    analogy = f"{a} is to {b} as {cc} is to {d}"
    lines = [
        "=" * 76,
        f"Sample {i} of {n}",
        f"Relation: {rel_name} ({rel_key})   |   triple_id: {tid}",
        f"Eligible: {elig}   |   Confidence: {conf}",
        f"completion_id: {cid}",
        "",
        f"  {analogy}",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional: also write the same text to this file.",
    )
    ap.add_argument("--n", type=int, default=50, help="How many completions to sample.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--include-ineligible",
        action="store_true",
        help="Allow ineligible participants (default: eligible only).",
    )
    args = ap.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Missing input: {args.input}")

    all_c = _load_completions(args.input)
    if args.include_ineligible:
        pool = all_c
    else:
        pool = [c for c in all_c if c.get("participant_eligible_complete") is True]

    if not pool:
        raise SystemExit("No completions in pool after filtering.")

    rng = random.Random(args.seed)
    k = min(args.n, len(pool))
    if k < args.n:
        print(
            f"Note: only {len(pool)} completions in pool; sampling all of them.",
            file=sys.stderr,
            flush=True,
        )
    chosen = rng.sample(pool, k=k)

    header = [
        "Human relational completions — random sample",
        f"source: {args.input}",
        f"seed: {args.seed}  |  n: {k}  |  eligible_only: {not args.include_ineligible}",
        "",
    ]
    body = [_format_block(i + 1, k, c) for i, c in enumerate(chosen)]
    text = "\n".join(header + body) + "\n" + "=" * 76 + "\n"

    sys.stdout.write(text)
    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")
        print(f"Also saved {k} samples to {args.out}", file=sys.stderr, flush=True)
    else:
        print(f"Printed {k} samples to stdout.", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
