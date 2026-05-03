from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description=(
            "Summarize relation-following rates from LLM_relations_completions "
            "trial NDJSON (judge_majority_correct / judge labels), by relation "
            "and by model."
        )
    )
    p.add_argument(
        "--trials-ndjson",
        type=Path,
        default=here.parent
        / "LLM_relations_completions"
        / "runs"
        / "gold_curate_b"
        / "trials.ndjson",
        help="Trial records (one JSON object per line).",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=here / "llm_validate_relation_following.json",
        help="Output summary path.",
    )
    p.add_argument(
        "--prompt-type",
        type=str,
        default=None,
        help=(
            "If set, keep only trials with this prompt_type (e.g. colon). "
            "Default: include all prompt types."
        ),
    )
    return p.parse_args()


def _relation_followed(rec: Dict[str, Any]) -> Optional[bool]:
    """True iff graders marked the completion as satisfying the relation."""
    if rec.get("error"):
        return None
    if "judge_majority_correct" in rec:
        return bool(rec["judge_majority_correct"])
    labels = rec.get("judge_labels")
    if not labels:
        return None
    if rec.get("judge_majority_label") in ("Correct", "Incorrect"):
        return rec["judge_majority_label"] == "Correct"
    return all(str(l).strip() == "Correct" for l in labels)


def _accumulate(
    trials_path: Path,
    prompt_type_filter: Optional[str],
) -> Tuple[
    Dict[str, str],
    DefaultDict[str, DefaultDict[str, List[int]]],
    Dict[str, int],
]:
    """
    Returns:
      relation_names: relation_key -> relation_name (first seen)
      counts: relation_key -> model -> [n_followed, n_total]
      skips: reason -> count
    """
    relation_names: Dict[str, str] = {}
    counts: DefaultDict[str, DefaultDict[str, List[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )
    skips: Dict[str, int] = defaultdict(int)

    with trials_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skips["json_decode_error"] += 1
                continue

            if prompt_type_filter is not None:
                if rec.get("prompt_type") != prompt_type_filter:
                    skips["prompt_type_filtered_out"] += 1
                    continue

            rel_key = rec.get("relation_key")
            model = rec.get("model")
            if not rel_key or not model:
                skips["missing_relation_key_or_model"] += 1
                continue

            if rel_key not in relation_names and rec.get("relation_name"):
                relation_names[rel_key] = str(rec["relation_name"])

            followed = _relation_followed(rec)
            if followed is None:
                skips["missing_or_failed_judgment"] += 1
                continue

            cell = counts[rel_key][str(model)]
            cell[1] += 1
            if followed:
                cell[0] += 1

    return relation_names, counts, dict(skips)


def _pct(numer: int, denom: int) -> float:
    return (numer / denom) if denom else 0.0


def _build_payload(
    trials_path: Path,
    prompt_type_filter: Optional[str],
    relation_names: Dict[str, str],
    counts: DefaultDict[str, DefaultDict[str, List[int]]],
    skips: Dict[str, int],
    n_lines_attempted: int,
) -> Dict[str, Any]:
    by_relation: Dict[str, Any] = {}
    model_totals: DefaultDict[str, List[int]] = defaultdict(lambda: [0, 0])

    for rel_key in sorted(counts.keys()):
        rel_name = relation_names.get(rel_key, rel_key)
        by_model: Dict[str, Any] = {}
        r_followed = 0
        r_total = 0
        for model in sorted(counts[rel_key].keys()):
            f_ct, t_ct = counts[rel_key][model]
            by_model[model] = {
                "n_trials": t_ct,
                "n_relation_followed": f_ct,
                "pct_relation_followed": _pct(f_ct, t_ct),
            }
            r_followed += f_ct
            r_total += t_ct
            mt = model_totals[model]
            mt[0] += f_ct
            mt[1] += t_ct
        by_relation[rel_key] = {
            "relation_key": rel_key,
            "relation_name": rel_name,
            "n_trials": r_total,
            "n_relation_followed": r_followed,
            "pct_relation_followed": _pct(r_followed, r_total),
            "by_model": by_model,
        }

    by_model_all: Dict[str, Any] = {}
    for model in sorted(model_totals.keys()):
        f_ct, t_ct = model_totals[model]
        by_model_all[model] = {
            "n_trials": t_ct,
            "n_relation_followed": f_ct,
            "pct_relation_followed": _pct(f_ct, t_ct),
            "by_relation": {
                rk: {
                    "n_trials": counts[rk][model][1],
                    "n_relation_followed": counts[rk][model][0],
                    "pct_relation_followed": _pct(
                        counts[rk][model][0], counts[rk][model][1]
                    ),
                }
                for rk in sorted(counts.keys())
                if model in counts[rk]
            },
        }

    scored = sum(
        counts[rk][m][1] for rk in counts for m in counts[rk]
    )

    return {
        "meta": {
            "trials_ndjson": str(trials_path.resolve()),
            "prompt_type_filter": prompt_type_filter,
            "relation_followed_definition": (
                "Trial counts as relation-followed when judge_majority_correct is "
                "true, or judge_majority_label == 'Correct', or all judge_labels "
                "are 'Correct'. Trials with non-null error or missing judgment "
                "are skipped."
            ),
            "n_lines_read": n_lines_attempted,
            "n_trials_scored": scored,
            "skip_counts": skips,
        },
        "by_relation": by_relation,
        "by_model": by_model_all,
    }


def _count_lines(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def main() -> None:
    args = parse_args()
    if not args.trials_ndjson.is_file():
        raise SystemExit(f"Trials file not found: {args.trials_ndjson}")

    n_lines = _count_lines(args.trials_ndjson)
    relation_names, counts, skips = _accumulate(
        args.trials_ndjson, args.prompt_type
    )
    payload = _build_payload(
        args.trials_ndjson,
        args.prompt_type,
        relation_names,
        counts,
        skips,
        n_lines,
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {args.out_json}")
    print(
        f"Scored {payload['meta']['n_trials_scored']} trials "
        f"({payload['meta']['n_lines_read']} non-empty lines)."
    )


if __name__ == "__main__":
    main()
