from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Print random triple outputs from humans and models."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=here / "gold_standard_humans_models.json",
        help="Path to linked gold-standard JSON.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=3,
        help="Number of random triples to print.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible sampling.",
    )
    return parser.parse_args()


def _format_humans(human_slots: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for slot in human_slots:
        pid = slot.get("participant_id")
        if not pid:
            continue
        response = slot.get("response")
        confidence = slot.get("confidence")
        lines.append(f"- {pid}: {response!r} (confidence={confidence!r})")
    if not lines:
        lines.append("- <no human responses>")
    return lines


def _format_models(model_slots: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for m in model_slots:
        model_name = m.get("model_name", "<unknown_model>")
        by_prompt = m.get("responses_by_prompt_type", {})
        prompt_bits: List[str] = []
        for prompt_type in sorted(by_prompt.keys()):
            d = by_prompt[prompt_type].get("D")
            prompt_bits.append(f"{prompt_type}={d!r}")
        lines.append(f"- {model_name}: " + ", ".join(prompt_bits))
    return lines


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    triples = payload.get("triples", [])
    if not triples:
        raise ValueError(f"No triples found in {args.input_json}")

    rng = random.Random(args.seed) if args.seed is not None else random.Random()
    k = min(args.n, len(triples))
    sampled = rng.sample(triples, k)

    print(f"Showing {k} random triples from {args.input_json}")
    for i, t in enumerate(sampled, start=1):
        tid = t.get("triple_id")
        relation_key = t.get("relation_key")
        a = t.get("A")
        b = t.get("B")
        c = t.get("C")
        print("\n" + "=" * 80)
        print(f"[{i}] triple_id={tid} relation={relation_key}")
        print(f"Prompt: {a} : {b} :: {c} : ____")

        print("\nHumans:")
        for line in _format_humans(t.get("human_slots", [])):
            print(line)

        print("\nModels:")
        for line in _format_models(t.get("model_slots", [])):
            print(line)
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
