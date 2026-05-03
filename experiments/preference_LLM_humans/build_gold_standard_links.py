from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    root = here.parent
    parser = argparse.ArgumentParser(
        description=(
            "Link kept human responses with model responses from gold_curate_b "
            "into a single per-triple gold-standard JSON."
        )
    )
    parser.add_argument(
        "--human-json",
        type=Path,
        default=root / "LLM_relations_completions" / "human_completions" / "human_answers_processed.json",
    )
    parser.add_argument(
        "--model-trials",
        type=Path,
        default=root / "LLM_relations_completions" / "runs" / "gold_curate_b" / "trials.ndjson",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=here / "gold_standard_humans_models.json",
    )
    return parser.parse_args()


def _load_humans(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_model_trials(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    human_payload = _load_humans(args.human_json)
    model_trials = _load_model_trials(args.model_trials)

    triples: Dict[int, Dict[str, Any]] = {}
    all_models = sorted({str(r["model"]) for r in model_trials})
    all_prompt_types = sorted({str(r["prompt_type"]) for r in model_trials})

    # Initialize triple rows from model trials (guaranteed full 500 coverage).
    for r in model_trials:
        tid = int(r["triple_id"])
        if tid not in triples:
            triples[tid] = {
                "triple_id": tid,
                "relation_key": r["relation_key"],
                "relation_name": r.get("relation_name"),
                "relation_text": r.get("relation_text"),
                "A": r["A"],
                "B": r["B"],
                "C": r["C"],
                "A_metrics": r.get("A_metrics"),
                "B_metrics": r.get("B_metrics"),
                "C_metrics": r.get("C_metrics"),
                "human_slots": [],
                "model_slots": [],
            }

    # Fill human slots (kept participants only), grouped by triple_id.
    humans_by_triple: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    participants = human_payload.get("participants", {})
    for response_id, p in participants.items():
        if not p.get("eligible_complete"):
            continue
        for item in p.get("items", []):
            tid = int(item["triple_id"])
            humans_by_triple[tid].append(
                {
                    "participant_id": response_id,
                    "prolific_pid": p.get("prolific_pid", ""),
                    "set_number": p.get("set_number"),
                    "item_number": item.get("item_number"),
                    "response": item.get("response", ""),
                    "confidence": item.get("confidence", ""),
                }
            )

    # Truncate/pad to 4 human slots per triple.
    for tid, row in triples.items():
        hs = humans_by_triple.get(tid, [])
        hs_sorted = sorted(
            hs,
            key=lambda x: (
                str(x.get("participant_id", "")),
                int(x.get("item_number") or 0),
            ),
        )
        row["human_slots"] = hs_sorted[:4]
        while len(row["human_slots"]) < 4:
            row["human_slots"].append(
                {
                    "participant_id": None,
                    "prolific_pid": None,
                    "set_number": None,
                    "item_number": None,
                    "response": None,
                    "confidence": None,
                }
            )

    # Fill model slots (one slot per model), store responses across prompt types.
    model_prompt_map: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    for r in model_trials:
        tid = int(r["triple_id"])
        model = str(r["model"])
        ptype = str(r["prompt_type"])
        model_prompt_map[tid][model][ptype] = {
            "D": r.get("D"),
            "D_metrics": r.get("D_metrics"),
            "grade_correct": r.get("grade_correct"),
            "judge_majority_label": r.get("judge_majority_label"),
            "judge_majority_correct": r.get("judge_majority_correct"),
            "judge_error": r.get("judge_error"),
            "error": r.get("error"),
            "raw_response_D": r.get("raw_response_D"),
            "prompt_complete": r.get("prompt_complete"),
            "timestamp_utc": r.get("timestamp_utc"),
        }

    for tid, row in triples.items():
        model_slots: List[Dict[str, Any]] = []
        for model in all_models:
            responses = model_prompt_map[tid].get(model, {})
            model_slots.append(
                {
                    "model_name": model,
                    "responses_by_prompt_type": {
                        p: responses.get(
                            p,
                            {
                                "D": None,
                                "D_metrics": None,
                                "grade_correct": None,
                                "judge_majority_label": None,
                                "judge_majority_correct": None,
                                "judge_error": None,
                                "error": None,
                                "raw_response_D": None,
                                "prompt_complete": None,
                                "timestamp_utc": None,
                            },
                        )
                        for p in all_prompt_types
                    },
                }
            )
        row["model_slots"] = model_slots

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "human_json": str(args.human_json.resolve()),
            "model_trials_ndjson": str(args.model_trials.resolve()),
        },
        "shape": {
            "n_triples": len(triples),
            "human_slots_per_triple": 4,
            "model_slots_per_triple": len(all_models),
            "prompt_types_per_model": all_prompt_types,
            "models": all_models,
        },
        "triples": [triples[k] for k in sorted(triples.keys())],
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote linked gold-standard JSON: {args.out_json}")


if __name__ == "__main__":
    main()
