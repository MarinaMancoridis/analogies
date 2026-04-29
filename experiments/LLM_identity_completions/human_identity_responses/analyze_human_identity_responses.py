from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Dict, List


CONFIDENCE_TO_SCORE = {
    "unsure": 1.0,
    "neutral": 2.0,
    "confident": 3.0,
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Analyze human identity responses by set. "
            "Keeps only complete IP Address submissions, normalizes answers "
            "(case/whitespace-insensitive), and computes agreement/confidence."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=here,
        help="Directory containing identity_*_final_human_validation.csv files.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=here / "identity_human_agreement_summary.json",
        help="Where to write the JSON summary.",
    )
    return parser.parse_args()


def _normalize_answer(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _get_loop_columns(fieldnames: List[str]) -> Dict[str, List[str]]:
    answer_cols = sorted(
        [c for c in fieldnames if re.match(r"^\d+_Q24$", c)],
        key=lambda c: int(c.split("_", 1)[0]),
    )
    confidence_cols = sorted(
        [c for c in fieldnames if re.match(r"^\d+_Q49$", c)],
        key=lambda c: int(c.split("_", 1)[0]),
    )
    if not answer_cols or len(answer_cols) != len(confidence_cols):
        raise ValueError("Could not find matching loop answer/confidence columns.")
    return {"answer_cols": answer_cols, "confidence_cols": confidence_cols}


def _is_complete_ip_row(row: Dict[str, str], answer_cols: List[str], confidence_cols: List[str]) -> bool:
    if (row.get("Status") or "").strip() != "IP Address":
        return False
    if (row.get("Finished") or "").strip().upper() != "TRUE":
        return False
    if (row.get("Progress") or "").strip() != "100":
        return False
    for col in answer_cols + confidence_cols:
        if not _normalize_answer(row.get(col, "")):
            return False
    return True


def analyze_file(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"No header found in {path}")
        cols = _get_loop_columns(reader.fieldnames)
        answer_cols = cols["answer_cols"]
        confidence_cols = cols["confidence_cols"]

        valid_rows: List[Dict[str, str]] = []
        for row in reader:
            if _is_complete_ip_row(row, answer_cols, confidence_cols):
                valid_rows.append(row)

    n_valid = len(valid_rows)
    n_triples = len(answer_cols)
    if n_valid == 0:
        return {
            "set_name": path.stem,
            "n_triples": n_triples,
            "n_valid_respondents": 0,
            "mean_correctness_pct": 0.0,
            "mean_majority_agreement_pct": 0.0,
            "p_agree_given_any_wrong_pct": 0.0,
            "p_agree_given_all_wrong_pct": 0.0,
            "n_any_wrong_triples": 0,
            "n_all_wrong_triples": 0,
            "mean_confidence_score_1_to_3": 0.0,
            "mean_confidence_pct": 0.0,
            "triple_stats": [],
        }

    triple_stats: List[Dict[str, object]] = []
    majority_rates: List[float] = []
    correctness_rates: List[float] = []
    confidence_scores: List[float] = []
    confidence_pcts: List[float] = []
    agree_count = 0
    any_wrong_count = 0
    agree_and_any_wrong_count = 0
    all_wrong_count = 0
    agree_and_all_wrong_count = 0

    for a_col, c_col in zip(answer_cols, confidence_cols):
        answer_counter: Counter[str] = Counter()
        conf_scores_for_triple: List[float] = []
        for row in valid_rows:
            norm_answer = _normalize_answer(row.get(a_col, ""))
            answer_counter[norm_answer] += 1
            conf_label = _normalize_answer(row.get(c_col, ""))
            conf_scores_for_triple.append(CONFIDENCE_TO_SCORE.get(conf_label, 0.0))

        majority_answer, majority_count = answer_counter.most_common(1)[0]
        majority_rate = majority_count / n_valid
        correctness_rate = majority_count / n_valid  # "correct" is defined as matching majority label
        avg_conf_score = mean(conf_scores_for_triple) if conf_scores_for_triple else 0.0
        avg_conf_pct = ((avg_conf_score - 1.0) / 2.0) * 100.0 if avg_conf_score > 0 else 0.0

        agree = (majority_count == n_valid)
        any_wrong = (majority_count < n_valid)
        # With majority-as-gold, all_wrong is structurally impossible.
        all_wrong = False

        if agree:
            agree_count += 1
        if any_wrong:
            any_wrong_count += 1
            if agree:
                agree_and_any_wrong_count += 1
        if all_wrong:
            all_wrong_count += 1
            if agree:
                agree_and_all_wrong_count += 1

        majority_rates.append(majority_rate)
        correctness_rates.append(correctness_rate)
        confidence_scores.append(avg_conf_score)
        confidence_pcts.append(avg_conf_pct)
        triple_stats.append(
            {
                "loop_index": int(a_col.split("_", 1)[0]),
                "majority_answer": majority_answer,
                "majority_count": majority_count,
                "correct_count": majority_count,
                "correctness_pct": round(correctness_rate * 100.0, 2),
                "n_wrong": n_valid - majority_count,
                "agreement_pct": round(majority_rate * 100.0, 2),
                "avg_confidence_score_1_to_3": round(avg_conf_score, 3),
                "avg_confidence_pct": round(avg_conf_pct, 2),
            }
        )

    p_agree_any_wrong = (agree_and_any_wrong_count / any_wrong_count) if any_wrong_count else 0.0
    p_agree_all_wrong = (agree_and_all_wrong_count / all_wrong_count) if all_wrong_count else 0.0

    return {
        "set_name": path.stem,
        "n_triples": n_triples,
        "n_valid_respondents": n_valid,
        "correctness_definition": "A response is correct iff it matches the majority label for that triple.",
        "mean_correctness_pct": round(mean(correctness_rates) * 100.0, 2),
        "mean_majority_agreement_pct": round(mean(majority_rates) * 100.0, 2),
        "p_agree_given_any_wrong_pct": round(p_agree_any_wrong * 100.0, 2),
        "p_agree_given_all_wrong_pct": round(p_agree_all_wrong * 100.0, 2),
        "n_any_wrong_triples": any_wrong_count,
        "n_all_wrong_triples": all_wrong_count,
        "mean_confidence_score_1_to_3": round(mean(confidence_scores), 3),
        "mean_confidence_pct": round(mean(confidence_pcts), 2),
        "triple_stats": triple_stats,
    }


def main() -> None:
    args = parse_args()
    files = sorted(args.input_dir.glob("identity_*_final_human_validation.csv"))
    if not files:
        raise FileNotFoundError(f"No identity_*_final_human_validation.csv files in {args.input_dir}")

    set_summaries = [analyze_file(p) for p in files]
    valid_sets = [s for s in set_summaries if int(s["n_valid_respondents"]) > 0]
    overall = {
        "correctness_definition": "A response is correct iff it matches the majority label for that triple.",
        "mean_correctness_pct": round(mean(float(s["mean_correctness_pct"]) for s in valid_sets), 2)
        if valid_sets
        else 0.0,
        "mean_majority_agreement_pct": round(
            mean(float(s["mean_majority_agreement_pct"]) for s in valid_sets), 2
        )
        if valid_sets
        else 0.0,
        "mean_confidence_pct": round(mean(float(s["mean_confidence_pct"]) for s in valid_sets), 2)
        if valid_sets
        else 0.0,
        "p_agree_given_any_wrong_pct": round(
            mean(float(s["p_agree_given_any_wrong_pct"]) for s in valid_sets), 2
        )
        if valid_sets
        else 0.0,
        "p_agree_given_all_wrong_pct": round(
            mean(float(s["p_agree_given_all_wrong_pct"]) for s in valid_sets), 2
        )
        if valid_sets
        else 0.0,
    }
    output = {
        "n_sets": len(set_summaries),
        "overall": overall,
        "sets": set_summaries,
    }
    args.out_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Found {len(set_summaries)} sets.")
    for s in set_summaries:
        print(
            f"{s['set_name']}: valid_respondents={s['n_valid_respondents']} "
            f"triples={s['n_triples']} "
            f"correctness={s['mean_correctness_pct']}% "
            f"agreement={s['mean_majority_agreement_pct']}% "
            f"confidence={s['mean_confidence_pct']}%"
        )
    print(f"Wrote summary: {args.out_json}")


if __name__ == "__main__":
    main()
