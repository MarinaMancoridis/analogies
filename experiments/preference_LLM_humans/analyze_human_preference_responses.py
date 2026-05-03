from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description=(
            "Summarize human analogy comparison survey: by Set_Number (group), "
            "how many eligible participants and aggregate rate of preferring the "
            "human-sourced completion. Links responses to qualtrics_loop_and_merge "
            "preference_set_*.csv for human_is_completion."
        )
    )
    p.add_argument(
        "--responses-csv",
        type=Path,
        default=here / "human_analogy_comparisons.csv",
        help="Qualtrics export CSV.",
    )
    p.add_argument(
        "--preference-dir",
        type=Path,
        default=here / "qualtrics_loop_and_merge",
        help="Directory with preference_set_01.csv … preference_set_NN.csv.",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=here / "human_preference_analysis.json",
        help="Output JSON path.",
    )
    p.add_argument(
        "--q23-correct",
        type=str,
        default="Completion A: petal",
        help="Exact expected text for comprehension Q23 (edit if survey text changes).",
    )
    p.add_argument(
        "--q52-correct",
        type=str,
        default=(
            "You may select which completion you prefer, and independently indicate "
            "which completions are valid."
        ),
        help="Exact expected text for comprehension Q52.",
    )
    p.add_argument(
        "--n-items",
        type=int,
        default=25,
        help="Number of comparison items per participant (must match preference_set rows).",
    )
    return p.parse_args()


def _load_headers_and_rows(csv_path: Path) -> Tuple[List[str], List[List[str]]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)
        next(reader)  # Qualtrics question-text row
        rows = list(reader)
    return headers, rows


def _choice_col_indices(headers: List[str]) -> Dict[Tuple[int, int], int]:
    """Map (item_order, set_number) -> column index for '{item}_Group {set} Q'."""
    pat = re.compile(r"^(\d+)_Group (\d+) Q$")
    out: Dict[Tuple[int, int], int] = {}
    for j, h in enumerate(headers):
        m = pat.match(h)
        if m:
            item = int(m.group(1))
            g = int(m.group(2))
            out[(item, g)] = j
    return out


def _load_preference_set(pref_dir: Path, set_number: int) -> Dict[int, Dict[str, Any]]:
    path = pref_dir / f"preference_set_{set_number:02d}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    by_item: Dict[int, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            io = int(row["Item_Order"])
            by_item[io] = row
    return by_item


def _parse_completion_choice(cell: str) -> Optional[int]:
    s = (cell or "").strip()
    if s == "Completion A":
        return 1
    if s == "Completion B":
        return 2
    return None


def _row_get(row: List[str], idx: int) -> str:
    if idx >= len(row):
        return ""
    return row[idx]


def _fleiss_kappa_binary(ratings: List[List[bool]]) -> Optional[Dict[str, float]]:
    """Fleiss' kappa for k=2 categories; each inner list is one rater (length n_items)."""
    if len(ratings) < 2:
        return None
    n_items = len(ratings[0])
    n_raters = len(ratings)
    if any(len(r) != n_items for r in ratings):
        return None
    p_sum = 0.0
    for i in range(n_items):
        n1 = sum(1 for r in ratings if r[i])
        n0 = n_raters - n1
        p_i = (n0 * (n0 - 1) + n1 * (n1 - 1)) / (n_raters * (n_raters - 1))
        p_sum += p_i
    p_bar = p_sum / n_items

    tot0 = sum(sum(1 for r in ratings if not r[i]) for i in range(n_items))
    tot1 = n_items * n_raters - tot0
    p0 = tot0 / (n_items * n_raters)
    p1 = tot1 / (n_items * n_raters)
    p_e = p0 * p0 + p1 * p1
    denom = 1.0 - p_e
    if denom <= 1e-12:
        kappa = 1.0 if math.isclose(p_bar, p_e, abs_tol=1e-9) else float("nan")
    else:
        kappa = (p_bar - p_e) / denom
    return {"fleiss_kappa": kappa, "P_bar": p_bar, "P_e": p_e}


def _mean_pairwise_cohens_kappa(ratings: List[List[bool]]) -> Optional[float]:
    if len(ratings) < 2:
        return None
    n_items = len(ratings[0])
    kappas: List[float] = []
    for idx_a, idx_b in itertools.combinations(range(len(ratings)), 2):
        ra = ratings[idx_a]
        rb = ratings[idx_b]
        agree = sum(1 for i in range(n_items) if ra[i] == rb[i])
        po = agree / n_items
        p_a1 = sum(1 for i in range(n_items) if ra[i]) / n_items
        p_b1 = sum(1 for i in range(n_items) if rb[i]) / n_items
        pe = p_a1 * p_b1 + (1.0 - p_a1) * (1.0 - p_b1)
        denom = 1.0 - pe
        if denom <= 1e-12:
            kappas.append(1.0 if math.isclose(po, pe, abs_tol=1e-9) else 0.0)
        else:
            kappas.append((po - pe) / denom)
    return sum(kappas) / len(kappas)


def _item_agreement_counts(ratings: List[List[bool]]) -> Dict[str, int]:
    n_items = len(ratings[0])
    n_raters = len(ratings)
    unanimous = 0
    for i in range(n_items):
        n1 = sum(1 for r in ratings if r[i])
        if n1 == 0 or n1 == n_raters:
            unanimous += 1
    return {
        "items_unanimous": unanimous,
        "items_with_disagreement": n_items - unanimous,
    }


def _agreement_block_for_group(
    set_number: int, ratings: List[List[bool]], n_items: int
) -> Dict[str, Any]:
    n_raters = len(ratings)
    base: Dict[str, Any] = {
        "set_number": set_number,
        "n_raters": n_raters,
        "n_items": n_items,
    }
    if n_raters < 2:
        base["fleiss_kappa"] = None
        base["mean_pairwise_cohens_kappa"] = None
        base["observed_agreement_P_bar"] = None
        base["expected_agreement_P_e"] = None
        base["note"] = "Fleiss' kappa and pairwise Cohen's kappa require at least 2 raters."
        if n_raters == 1:
            base.update(_item_agreement_counts(ratings))
        else:
            base["items_unanimous"] = 0
            base["items_with_disagreement"] = 0
        return base

    fk = _fleiss_kappa_binary(ratings)
    assert fk is not None
    kappa_val = fk["fleiss_kappa"]
    out = {
        **base,
        "fleiss_kappa": None if isinstance(kappa_val, float) and math.isnan(kappa_val) else kappa_val,
        "mean_pairwise_cohens_kappa": _mean_pairwise_cohens_kappa(ratings),
        "observed_agreement_P_bar": fk["P_bar"],
        "expected_agreement_P_e": fk["P_e"],
        **_item_agreement_counts(ratings),
    }
    return out


def main() -> None:
    args = parse_args()
    headers, rows = _load_headers_and_rows(args.responses_csv)
    choice_cols = _choice_col_indices(headers)

    try:
        idx_status = headers.index("Status")
        idx_progress = headers.index("Progress")
        idx_finished = headers.index("Finished")
        idx_response = headers.index("ResponseId")
        idx_q23 = headers.index("Q23")
        idx_q52 = headers.index("Q52")
        idx_set = headers.index("Set_Number")
    except ValueError as e:
        raise SystemExit(f"Missing expected column in CSV: {e}") from e

    max_set = 0
    for (_, g) in choice_cols:
        max_set = max(max_set, g)
    n_groups = max_set

    pref_cache: Dict[int, Dict[int, Dict[str, Any]]] = {}

    def pref_for(set_num: int) -> Dict[int, Dict[str, Any]]:
        if set_num not in pref_cache:
            pref_cache[set_num] = _load_preference_set(args.preference_dir, set_num)
        return pref_cache[set_num]

    participants_out: List[Dict[str, Any]] = []
    by_set_ratings: Dict[int, List[List[bool]]] = defaultdict(list)
    group_stats: Dict[int, Dict[str, Any]] = {
        g: {
            "set_number": g,
            "n_participants_eligible": 0,
            "n_judgments": 0,
            "n_prefer_human_completion": 0,
            "participant_ids": [],
        }
        for g in range(1, n_groups + 1)
    }

    screening = {
        "total_rows": len(rows),
        "ip_address_rows": 0,
        "eligible_complete": 0,
        "exclusion_counts": defaultdict(int),
    }

    for row in rows:
        status = _row_get(row, idx_status)
        if status != "IP Address":
            continue
        screening["ip_address_rows"] += 1
        rid = _row_get(row, idx_response)
        progress = _row_get(row, idx_progress).strip()
        finished = _row_get(row, idx_finished).strip().upper()
        q23 = _row_get(row, idx_q23).strip()
        q52 = _row_get(row, idx_q52).strip()
        set_raw = _row_get(row, idx_set).strip()

        def exclude(reason: str) -> None:
            screening["exclusion_counts"][reason] += 1
            participants_out.append(
                {
                    "response_id": rid,
                    "eligible": False,
                    "exclusion_reason": reason,
                    "set_number": set_raw,
                }
            )

        if progress != "100":
            exclude("progress_not_100")
            continue
        if finished != "TRUE":
            exclude("not_finished")
            continue
        if not set_raw.isdigit():
            exclude("set_number_missing_or_invalid")
            continue
        set_num = int(set_raw)
        if set_num < 1 or set_num > n_groups:
            exclude("set_number_out_of_range")
            continue
        if q23 != args.q23_correct:
            exclude("q23_incorrect")
            continue
        if q52 != args.q52_correct:
            exclude("q52_incorrect")
            continue

        # Optional: verify only this group's block has answers (spot data issues).
        filled_groups = []
        for g in range(1, n_groups + 1):
            nfill = 0
            for item in range(1, args.n_items + 1):
                key = (item, g)
                if key not in choice_cols:
                    continue
                j = choice_cols[key]
                if _row_get(row, j).strip():
                    nfill += 1
            if nfill:
                filled_groups.append((g, nfill))
        if len(filled_groups) != 1 or filled_groups[0][0] != set_num:
            screening["exclusion_counts"]["group_block_mismatch_or_multi_group_filled"] += 1
            participants_out.append(
                {
                    "response_id": rid,
                    "eligible": False,
                    "exclusion_reason": "group_block_mismatch_or_multi_group_filled",
                    "set_number": set_num,
                    "filled_groups": filled_groups,
                }
            )
            continue

        meta_by_item = pref_for(set_num)
        n_prefer = 0
        n_scored = 0
        missing_item = False
        per_item: List[Optional[bool]] = []

        for item in range(1, args.n_items + 1):
            key = (item, set_num)
            if key not in choice_cols:
                missing_item = True
                break
            col_j = choice_cols[key]
            choice = _parse_completion_choice(_row_get(row, col_j))
            if choice is None:
                missing_item = True
                per_item.append(None)
                continue
            if item not in meta_by_item:
                missing_item = True
                break
            human_is = int(meta_by_item[item]["human_is_completion"])
            prefer_human = choice == human_is
            per_item.append(prefer_human)
            n_scored += 1
            if prefer_human:
                n_prefer += 1

        if missing_item or n_scored != args.n_items:
            exclude("incomplete_comparison_items")
            continue

        screening["eligible_complete"] += 1
        pct_i = n_prefer / n_scored if n_scored else 0.0
        participants_out.append(
            {
                "response_id": rid,
                "eligible": True,
                "set_number": set_num,
                "n_items_scored": n_scored,
                "n_prefer_human_completion": n_prefer,
                "pct_prefer_human_completion": pct_i,
                "per_item_prefer_human": per_item,
            }
        )
        by_set_ratings[set_num].append([bool(x) for x in per_item])

        gs = group_stats[set_num]
        gs["n_participants_eligible"] += 1
        gs["n_judgments"] += n_scored
        gs["n_prefer_human_completion"] += n_prefer
        gs["participant_ids"].append(rid)

    groups_list: List[Dict[str, Any]] = []
    total_eligible = 0
    total_j = 0
    total_ph = 0
    for g in range(1, n_groups + 1):
        gs = group_stats[g]
        ne = gs["n_participants_eligible"]
        nj = gs["n_judgments"]
        nph = gs["n_prefer_human_completion"]
        total_eligible += ne
        total_j += nj
        total_ph += nph
        pct = (nph / nj) if nj else 0.0
        groups_list.append(
            {
                "set_number": g,
                "n_participants_eligible": ne,
                "n_judgments": nj,
                "n_prefer_human_completion": nph,
                "pct_prefer_human_completion": pct,
                "response_ids": gs["participant_ids"],
            }
        )

    agreement_by_group = [
        _agreement_block_for_group(g, by_set_ratings.get(g, []), args.n_items)
        for g in range(1, n_groups + 1)
    ]
    fleiss_values = [
        b["fleiss_kappa"]
        for b in agreement_by_group
        if b.get("fleiss_kappa") is not None
    ]
    cohen_values = [
        b["mean_pairwise_cohens_kappa"]
        for b in agreement_by_group
        if b.get("mean_pairwise_cohens_kappa") is not None
    ]
    agreement_overall = {
        "n_groups_with_at_least_2_raters": sum(
            1 for b in agreement_by_group if b["n_raters"] >= 2
        ),
        "mean_fleiss_kappa_unweighted": (
            sum(fleiss_values) / len(fleiss_values) if fleiss_values else None
        ),
        "mean_pairwise_cohens_kappa_unweighted": (
            sum(cohen_values) / len(cohen_values) if cohen_values else None
        ),
    }

    payload: Dict[str, Any] = {
        "n_groups": n_groups,
        "n_items_per_participant": args.n_items,
        "comprehension": {
            "q23_expected": args.q23_correct,
            "q52_expected": args.q52_correct,
        },
        "screening": dict(screening),
        "groups": groups_list,
        "overall": {
            "n_participants_eligible": total_eligible,
            "n_judgments": total_j,
            "n_prefer_human_completion": total_ph,
            "pct_prefer_human_completion": (total_ph / total_j) if total_j else 0.0,
        },
        "agreement": {
            "rating_definition": (
                "Each rater's response is prefer_human_completion (True) vs prefer_llm "
                "(False) per item, after mapping A/B using human_is_completion."
            ),
            "by_group": agreement_by_group,
            "overall_across_groups": agreement_overall,
        },
        "participants": participants_out,
        "notes": [
            "Set_Number (embedded data) must match the only filled Group block "
            "(e.g. Group 9 Q columns when Set_Number=9).",
            f"Only item orders 1..{args.n_items} are scored; Qualtrics may have "
            "an extra 26th item column not present in preference_set_*.csv.",
            "If you change survey wording, pass --q23-correct and --q52-correct.",
            "Fleiss' kappa is computed per Set_Number (same 25 items, constant N raters); "
            "it is not a single kappa over different item pools. "
            "mean_fleiss_kappa_unweighted is the unweighted mean of group kappas (N≥2).",
        ],
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {args.out_json}")
    mf = agreement_overall["mean_fleiss_kappa_unweighted"]
    mf_s = f"{mf:.4f}" if mf is not None else "n/a"
    print(
        f"Eligible participants: {total_eligible} | "
        f"Overall P(prefer human): {payload['overall']['pct_prefer_human_completion']:.4f} | "
        f"Mean Fleiss κ (by group): {mf_s}"
    )


if __name__ == "__main__":
    main()
