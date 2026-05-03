from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Build Qualtrics Loop & Merge CSV sets for preference questions."
    )
    parser.add_argument(
        "--predetermined-json",
        type=Path,
        default=here / "predetermined_pairwise_questions.json",
        help="Input predetermined pairwise question pool JSON.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=here / "qualtrics_loop_and_merge",
        help="Output directory for per-set CSVs.",
    )
    parser.add_argument(
        "--num-sets",
        type=int,
        default=10,
        help="Number of CSV sets to generate.",
    )
    parser.add_argument(
        "--questions-per-set",
        type=int,
        default=25,
        help="Number of preference questions per set CSV.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Seed for deterministic shuffling/sampling.",
    )
    parser.add_argument(
        "--required-llm-prompt-type",
        type=str,
        default="colon",
        help=(
            "Only include questions whose llm_source.prompt_type matches. "
            "Default colon matches the colon-only pairwise judging setup. "
            "Use zero_shot for {'colon','english'}."
        ),
    )
    return parser.parse_args()


def _question_row(set_number: int, item_order: int, q: Dict[str, Any]) -> Dict[str, Any]:
    llm_prompt_type = q["llm_source"]["prompt_type"]
    llm_prompt_family = "zero_shot" if llm_prompt_type in {"colon", "english"} else llm_prompt_type
    return {
        "Set_Number": set_number,
        "Item_Order": item_order,
        "comparison_id": q["comparison_id"],
        "triple_id": q["triple_id"],
        "analogy_prompt": f"{q['A']} : {q['B']} :: {q['C']} : ____",
        "completion_1": q["completion_1_text"],
        "completion_2": q["completion_2_text"],
        "completion_1_source": q["completion_1_source"],
        "completion_2_source": q["completion_2_source"],
        # Minimal linkage keys for reconstructing full metadata later.
        "human_is_completion": 1 if q["completion_1_source"] == "human" else 2,
        "human_participant_id": q["human_source"]["participant_id"],
        "llm_model_name": q["llm_source"]["model_name"],
        "llm_prompt_type": llm_prompt_type,
        "llm_prompt_family": llm_prompt_family,
    }


def main() -> None:
    args = parse_args()
    payload = json.loads(args.predetermined_json.read_text(encoding="utf-8"))
    questions: List[Dict[str, Any]] = payload["questions"]

    if args.required_llm_prompt_type:
        required = args.required_llm_prompt_type.strip()
        if required == "zero_shot":
            allowed = {"colon", "english"}
            questions = [
                q
                for q in questions
                if q.get("llm_source", {}).get("prompt_type") in allowed
            ]
        else:
            questions = [
                q
                for q in questions
                if q.get("llm_source", {}).get("prompt_type") == required
            ]
        print(f"Filtered to llm_source.prompt_type='{required}': {len(questions)} questions")

    total_needed = args.num_sets * args.questions_per_set
    if not questions:
        raise ValueError("No questions available after filtering.")

    rng = random.Random(args.seed)
    if len(questions) >= total_needed:
        selected = rng.sample(questions, total_needed)
    else:
        print(
            f"Only {len(questions)} questions available for requested constraints; "
            "sampling with replacement to fill all sets."
        )
        selected = [rng.choice(questions) for _ in range(total_needed)]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for set_idx in range(args.num_sets):
        start = set_idx * args.questions_per_set
        end = start + args.questions_per_set
        subset = selected[start:end]
        set_number = set_idx + 1

        rows = [_question_row(set_number, i + 1, q) for i, q in enumerate(subset)]
        out_csv = args.out_dir / f"preference_set_{set_number:02d}.csv"
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
