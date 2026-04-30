from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Process Qualtrics human answers for relational analogies. "
            "Validates comprehension + full completion for assigned Set_Number, "
            "prints per-set completion counts, and writes structured JSON output."
        )
    )
    parser.add_argument(
        "--answers-csv",
        type=Path,
        default=here / "humans.csv",
        help="Path to Qualtrics export CSV.",
    )
    parser.add_argument(
        "--blocks-dir",
        type=Path,
        default=here.parent / "qualtrics_loop_and_merge" / "curate_b_25_blocks",
        help="Directory containing curate_b_block_XX.csv files.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=here / "human_answers_processed.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


GROUP_QUESTION_RE = re.compile(r"^(\d+)_Group (\d+) Q$")
CONF_RE = re.compile(r"^(\d+)_Q(\d+)$")


def _norm(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _is_true_like(text: Any) -> bool:
    t = _norm(text)
    return t in {"true", "1", "yes"}


def _is_human_response_row(row: Dict[str, str]) -> bool:
    # Qualtrics exports often include extra metadata/header-ish rows after headers.
    if _norm(row.get("Status")) in {"", "status", "response type"}:
        return False
    if _norm(row.get("ResponseId")).startswith("{\"importid\""):
        return False
    return True


def _load_blocks(blocks_dir: Path) -> Dict[int, List[Dict[str, str]]]:
    blocks: Dict[int, List[Dict[str, str]]] = {}
    for path in sorted(blocks_dir.glob("curate_b_block_*.csv")):
        m = re.search(r"curate_b_block_(\d+)\.csv$", path.name)
        if not m:
            continue
        set_num = int(m.group(1))
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if len(rows) != 20:
            raise ValueError(f"{path} has {len(rows)} rows (expected 20)")
        blocks[set_num] = rows
    if len(blocks) != 25:
        raise ValueError(f"Expected 25 block files in {blocks_dir}, found {len(blocks)}")
    return blocks


def _discover_group_columns(fieldnames: List[str]) -> Dict[int, Dict[int, Tuple[str, str]]]:
    """
    Returns:
      group_to_item_map[group_num][item_num] = (answer_col, confidence_col)
    """
    group_to_item: Dict[int, Dict[int, Tuple[str, str]]] = defaultdict(dict)
    for i, col in enumerate(fieldnames):
        m = GROUP_QUESTION_RE.match(col)
        if not m:
            continue
        item_num = int(m.group(1))
        group_num = int(m.group(2))

        conf_col: Optional[str] = None
        # In this export, confidence column is immediately after answer column.
        if i + 1 < len(fieldnames):
            nxt = fieldnames[i + 1]
            m_conf = CONF_RE.match(nxt)
            if m_conf and int(m_conf.group(1)) == item_num:
                conf_col = nxt
        if conf_col is None:
            raise ValueError(f"Could not find confidence column paired with {col}")

        group_to_item[group_num][item_num] = (col, conf_col)

    # Ensure we have 25 groups x 20 items.
    if len(group_to_item) != 25:
        raise ValueError(f"Expected 25 groups in columns, found {len(group_to_item)}")
    for g in range(1, 26):
        if g not in group_to_item:
            raise ValueError(f"Missing Group {g} columns")
        if len(group_to_item[g]) != 20:
            raise ValueError(f"Group {g} has {len(group_to_item[g])} items (expected 20)")
    return group_to_item


def main() -> None:
    args = parse_args()
    blocks = _load_blocks(args.blocks_dir)

    with args.answers_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        group_cols = _discover_group_columns(fieldnames)
        raw_rows = [r for r in reader if _is_human_response_row(r)]

    per_set_counts = {s: {"total_rows": 0, "eligible_complete": 0} for s in range(1, 26)}
    participants: Dict[str, Dict[str, Any]] = {}

    for row in raw_rows:
        response_id = str(row.get("ResponseId") or "")
        set_raw = _norm(row.get("Set_Number"))
        set_number = int(set_raw) if set_raw.isdigit() else None

        status_ok = _norm(row.get("Status")) == "ip address"
        finished_ok = _is_true_like(row.get("Finished"))
        progress_ok = _norm(row.get("Progress")) == "100"
        comprehension_q23_ok = _norm(row.get("Q23")) == "petal"
        comprehension_q52_ok = _norm(row.get("Q52")) == "flower"
        comprehension_ok = comprehension_q23_ok and comprehension_q52_ok

        set_valid = set_number is not None and 1 <= set_number <= 25
        complete_group_questions = False
        items: List[Dict[str, Any]] = []

        if set_valid:
            per_set_counts[set_number]["total_rows"] += 1
            item_map = group_cols[set_number]
            block_rows = blocks[set_number]
            complete_group_questions = True
            for item_num in range(1, 21):
                ans_col, conf_col = item_map[item_num]
                answer = str(row.get(ans_col) or "")
                confidence = str(row.get(conf_col) or "")
                if _norm(answer) == "" or _norm(confidence) == "":
                    complete_group_questions = False

                triple_meta = block_rows[item_num - 1]
                items.append(
                    {
                        "item_number": item_num,
                        "triple_id": int(triple_meta["triple_id"]),
                        "relation_key": triple_meta["relation_key"],
                        "A": triple_meta["A"],
                        "B": triple_meta["B"],
                        "C": triple_meta["C"],
                        "response": answer,
                        "confidence": confidence,
                    }
                )

        eligible_complete = (
            set_valid
            and status_ok
            and finished_ok
            and progress_ok
            and comprehension_ok
            and complete_group_questions
        )
        if eligible_complete and set_number is not None:
            per_set_counts[set_number]["eligible_complete"] += 1

        participants[response_id] = {
            "response_id": response_id,
            "prolific_pid": row.get("PROLIFIC_PID", ""),
            "set_number": set_number,
            "status_ok": status_ok,
            "finished_ok": finished_ok,
            "progress_ok": progress_ok,
            "comprehension_q23_ok": comprehension_q23_ok,
            "comprehension_q52_ok": comprehension_q52_ok,
            "comprehension_ok": comprehension_ok,
            "complete_group_questions": complete_group_questions,
            "eligible_complete": eligible_complete,
            "items": items,
        }

    # Human-readable summary printout.
    print("=== Eligible complete participants by Set_Number ===")
    for s in range(1, 26):
        c = per_set_counts[s]
        print(
            f"Set {s:02d}: eligible_complete={c['eligible_complete']} / total_rows_with_set={c['total_rows']}"
        )

    kept_participants = [p for p in participants.values() if p["eligible_complete"]]
    total_analogies_filled_by_kept = sum(len(p["items"]) for p in kept_participants)

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_answers_csv": str(args.answers_csv.resolve()),
        "source_blocks_dir": str(args.blocks_dir.resolve()),
        "eligibility_rules": {
            "status": "IP Address",
            "finished": True,
            "progress": 100,
            "comprehension_q23": "petal",
            "comprehension_q52": "flower",
            "all_20_answer_and_confidence_filled_for_assigned_set": True,
        },
        "kept_summary": {
            "n_kept_participants": len(kept_participants),
            "total_analogies_filled_by_kept_participants": total_analogies_filled_by_kept,
        },
        "per_set_counts": per_set_counts,
        "participants": participants,
    }
    args.out_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote processed output: {args.out_json}")


if __name__ == "__main__":
    main()
