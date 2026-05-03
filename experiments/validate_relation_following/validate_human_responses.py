from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _load_process_human_answers_module():
    repo_exp = Path(__file__).resolve().parent.parent
    path = repo_exp / "LLM_relations_completions" / "human_completions" / "process_human_answers.py"
    if not path.is_file():
        raise FileNotFoundError(f"Expected process_human_answers at {path}")
    spec = importlib.util.spec_from_file_location("process_human_answers", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _relation_display_name(relation_key: str) -> str:
    return {
        "attribute": "Attribute",
        "case_relations": "Case Relations",
        "cause_purpose": "Cause-Purpose",
        "class_inclusion": "Class-Inclusion",
        "contrast": "Contrast",
        "non_attribute": "Non-Attribute",
        "part_whole": "Part-Whole",
        "reference": "Reference",
        "similar": "Similar",
        "space_time": "Space-Time",
    }.get(relation_key, relation_key.replace("_", " ").title())


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    hc = here.parent / "LLM_relations_completions" / "human_completions"
    p = argparse.ArgumentParser(
        description=(
            "Flatten Qualtrics human analogy completions into one JSON record per "
            "completion (A, B, C, human D, relation, metadata). Uses the same "
            "blocks and eligibility logic as process_human_answers.py."
        )
    )
    p.add_argument(
        "--answers-csv",
        type=Path,
        default=hc / "humans.csv",
        help="Qualtrics export (same as process_human_answers).",
    )
    p.add_argument(
        "--blocks-dir",
        type=Path,
        default=here.parent
        / "LLM_relations_completions"
        / "qualtrics_loop_and_merge"
        / "curate_b_25_blocks",
        help="curate_b_block_XX.csv directory.",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=here / "human_completions_flat.json",
        help="Output: list of completion records.",
    )
    p.add_argument(
        "--only-eligible",
        action="store_true",
        help="Keep only rows where participant_eligible_complete is true.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ph = _load_process_human_answers_module()

    blocks = ph._load_blocks(args.blocks_dir)
    with args.answers_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        group_cols = ph._discover_group_columns(fieldnames)
        raw_rows = [r for r in reader if ph._is_human_response_row(r)]

    completions: List[Dict[str, Any]] = []

    for row in raw_rows:
        response_id = str(row.get("ResponseId") or "")
        set_raw = ph._norm(row.get("Set_Number"))
        set_number = int(set_raw) if set_raw.isdigit() else None

        status_ok = ph._norm(row.get("Status")) == "ip address"
        finished_ok = ph._is_true_like(row.get("Finished"))
        progress_ok = ph._norm(row.get("Progress")) == "100"
        comprehension_q23_ok = ph._norm(row.get("Q23")) == "petal"
        comprehension_q52_ok = ph._norm(row.get("Q52")) == "flower"
        comprehension_ok = comprehension_q23_ok and comprehension_q52_ok

        set_valid = set_number is not None and 1 <= set_number <= 25
        complete_group_questions = False
        item_map: Dict[int, Any] = {}
        block_rows: List[Dict[str, str]] = []

        if set_valid:
            item_map = group_cols[set_number]
            block_rows = blocks[set_number]
            complete_group_questions = True
            for item_num in range(1, 21):
                ans_col, conf_col = item_map[item_num]
                answer = str(row.get(ans_col) or "")
                confidence = str(row.get(conf_col) or "")
                if ph._norm(answer) == "" or ph._norm(confidence) == "":
                    complete_group_questions = False

        eligible_complete = (
            set_valid
            and status_ok
            and finished_ok
            and progress_ok
            and comprehension_ok
            and complete_group_questions
        )

        if not set_valid:
            continue

        prolific_pid = str(row.get("PROLIFIC_PID") or "")

        for item_num in range(1, 21):
            ans_col, conf_col = item_map[item_num]
            answer = str(row.get(ans_col) or "").strip()
            confidence = str(row.get(conf_col) or "").strip()
            triple_meta = block_rows[item_num - 1]
            relation_key = str(triple_meta["relation_key"])

            rec = {
                "completion_id": f"{response_id}:set{set_number}:item{item_num}",
                "response_id": response_id,
                "prolific_pid": prolific_pid,
                "set_number": set_number,
                "item_number": item_num,
                "triple_id": int(triple_meta["triple_id"]),
                "relation_key": relation_key,
                "relation_name": _relation_display_name(relation_key),
                "A": str(triple_meta["A"]).strip(),
                "B": str(triple_meta["B"]).strip(),
                "C": str(triple_meta["C"]).strip(),
                "D": answer,
                "confidence": confidence,
                "participant_eligible_complete": eligible_complete,
            }
            if args.only_eligible and not eligible_complete:
                continue
            completions.append(rec)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "only_eligible_participants": args.only_eligible,
        "source_answers_csv": str(args.answers_csv.resolve()),
        "source_blocks_dir": str(args.blocks_dir.resolve()),
        "eligibility_note": (
            "participant_eligible_complete matches process_human_answers.py "
            "(IP Address, finished, 100%, Q23=petal, Q52=flower, all 20 answers "
            "and confidence fields filled)."
        ),
        "n_completions": len(completions),
        "n_eligible_completions": sum(
            1 for c in completions if c["participant_eligible_complete"]
        ),
        "completions": completions,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {args.out_json} ({payload['n_completions']} completions)")


if __name__ == "__main__":
    main()
