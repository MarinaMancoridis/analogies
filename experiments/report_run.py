# analogies/experiments/report_run.py
# usage
# python -m analogies.experiments.report_run analogies/experiments/runs/20260223-153012-a1b2c3
from __future__ import annotations

import os
import json
import argparse
import statistics
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple


# -----------------------
# IO helpers
# -----------------------

def read_ndjson(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


# -----------------------
# Generic "hit" inference
# -----------------------

HIT_KEYS = ("is_identity", "is_success", "is_correct", "grade_correct", "hit", "success")

def infer_hit(trial: Dict[str, Any]) -> Optional[bool]:
    for k in HIT_KEYS:
        if k in trial:
            try:
                return bool(trial[k])
            except Exception:
                return None
    return None


# -----------------------
# Formatting helpers
# -----------------------

def pct(x: float) -> str:
    return f"{100.0 * x:5.1f}%"

def safe_mean(xs: List[float]) -> Optional[float]:
    xs2 = [x for x in xs if x is not None]
    if not xs2:
        return None
    return float(statistics.mean(xs2))

def safe_quantiles(xs: List[float], qs=(0.1, 0.5, 0.9)) -> Optional[Tuple[float, ...]]:
    xs2 = sorted([x for x in xs if x is not None])
    if not xs2:
        return None
    def q(p: float) -> float:
        if len(xs2) == 1:
            return float(xs2[0])
        idx = p * (len(xs2) - 1)
        lo = int(idx)
        hi = min(lo + 1, len(xs2) - 1)
        frac = idx - lo
        return float(xs2[lo] * (1 - frac) + xs2[hi] * frac)
    return tuple(q(p) for p in qs)

def print_table(rows: List[List[str]], header: Optional[List[str]] = None) -> None:
    if header:
        rows = [header] + rows
    widths = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
    for i, r in enumerate(rows):
        line = "  ".join(r[c].ljust(widths[c]) for c in range(len(r)))
        print(line)
        if header and i == 0:
            print("  ".join("-" * w for w in widths))


# -----------------------
# Aggregations
# -----------------------

def group_trials_by_model(run_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Looks for subdirs inside run_dir that contain trials.ndjson.
    Returns model_dir_name -> trials
    """
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for name in sorted(os.listdir(run_dir)):
        p = os.path.join(run_dir, name)
        if not os.path.isdir(p):
            continue
        trials_path = os.path.join(p, "trials.ndjson")
        if os.path.exists(trials_path):
            by_model[name] = read_ndjson(trials_path)
    return by_model

def model_display_name(model_dir: str, trials: List[Dict[str, Any]]) -> str:
    # Prefer trial["model"] if present, else folder name
    for t in trials:
        m = t.get("model")
        if isinstance(m, str) and m.strip():
            return m
    return model_dir

def summarize_model(trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    hits: List[bool] = []
    for t in trials:
        h = infer_hit(t)
        if h is None:
            continue
        hits.append(h)

    n = len(hits)
    hit_rate = (sum(hits) / n) if n else None

    # Try to summarize A_mode/C_mode if present
    combo = Counter()
    for t in trials:
        am = t.get("A_mode")
        cm = t.get("C_mode")
        if isinstance(am, str) and isinstance(cm, str):
            combo[(am, cm)] += 1

    return {
        "n_scored": n,
        "hit_rate": hit_rate,
        "ac_combo_counts": combo,
    }

def distribution_summary(trials: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    vals: List[Any] = [t.get(key) for t in trials if key in t]
    # numeric?
    nums: List[float] = []
    for v in vals:
        if isinstance(v, (int, float)) and v is not None:
            nums.append(float(v))
    if nums:
        qs = safe_quantiles(nums)
        return {
            "type": "numeric",
            "count": len(nums),
            "mean": safe_mean(nums),
            "q10_q50_q90": qs,
        }
    # categorical
    cats = [v for v in vals if isinstance(v, str) and v]
    if cats:
        c = Counter(cats)
        top = c.most_common(8)
        return {
            "type": "categorical",
            "count": len(cats),
            "top": top,
        }
    return {"type": "none", "count": 0}

def collect_common_keys(by_model: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    key_counts = Counter()
    total_trials = 0
    for trials in by_model.values():
        for t in trials:
            total_trials += 1
            key_counts.update(t.keys())
    # keep keys that appear fairly often
    common = [k for k, c in key_counts.items() if c >= max(10, int(0.2 * total_trials))]
    # stable, useful-first ordering
    preferred = [
        "experiment", "analogy_type",
        "A_mode", "C_mode",
        "A_len", "C_len",
        "A_pop_zipf", "C_pop_zipf",
        "A_pop_rank", "C_pop_rank",
        "A_polysemy", "C_polysemy",
        "A_brys", "C_brys",
        "A_brys_label", "C_brys_label",
        "A_pos", "C_pos",
    ]
    ordered = [k for k in preferred if k in common]
    ordered += [k for k in sorted(common) if k not in set(ordered) and k not in {"prompt", "raw_response"}]
    return ordered

def print_error_examples(trials: List[Dict[str, Any]], max_examples: int = 8) -> None:
    """
    Prints a few failures with (A, C, parsed_answer, expected).
    """
    examples = []
    for t in trials:
        h = infer_hit(t)
        if h is False:
            A = t.get("A")
            C = t.get("C")
            pred = t.get("parsed_answer")
            exp = t.get("expected")
            if isinstance(A, str) and isinstance(C, str):
                examples.append((A, C, pred, exp))
        if len(examples) >= max_examples:
            break

    if not examples:
        print("  (no failure examples found)")
        return

    for i, (A, C, pred, exp) in enumerate(examples, 1):
        print(f"  {i:>2}. A={A!r}  C={C!r}  pred={pred!r}  expected={exp!r}")


# -----------------------
# Main report
# -----------------------

def report(run_dir: str, *, show_distributions: bool = True, show_errors: bool = True) -> None:
    run_dir = os.path.abspath(run_dir)
    print(f"\n=== Report: {run_dir} ===\n")

    run_summary_path = os.path.join(run_dir, "run_summary.json")
    if os.path.exists(run_summary_path):
        with open(run_summary_path, "r", encoding="utf-8") as f:
            run_summary = json.load(f)
        print("Run summary:")
        print(f"  run_id: {run_summary.get('run_id')}")
        print(f"  started_at: {run_summary.get('started_at')}")
        print(f"  ended_at: {run_summary.get('ended_at')}")
        print(f"  n_trials_per_model: {run_summary.get('n_trials_per_model')}")
        print(f"  freq_buckets: {run_summary.get('freq_buckets')}")
        print("")

    by_model = group_trials_by_model(run_dir)
    if not by_model:
        raise SystemExit(f"No model subdirs with trials.ndjson found under {run_dir}")

    # Leaderboard
    leaderboard_rows = []
    per_model_trials: List[Tuple[str, List[Dict[str, Any]]]] = []
    for model_dir, trials in by_model.items():
        name = model_display_name(model_dir, trials)
        summ = summarize_model(trials)
        n = summ["n_scored"]
        hr = summ["hit_rate"]
        leaderboard_rows.append([
            name,
            str(len(trials)),
            str(n),
            (pct(hr) if hr is not None else "n/a"),
        ])
        per_model_trials.append((name, trials))

    leaderboard_rows.sort(key=lambda r: float(r[3].strip("%")) if r[3] != "n/a" else -1.0, reverse=True)

    print("Leaderboard (sorted by hit rate):")
    print_table(
        leaderboard_rows,
        header=["model", "rows_in_file", "scored_trials", "hit_rate"]
    )
    print("")

    # A_mode/C_mode breakdown (if present)
    print("A_mode/C_mode mix (counts):")
    for name, trials in per_model_trials:
        combo = Counter()
        for t in trials:
            am, cm = t.get("A_mode"), t.get("C_mode")
            if isinstance(am, str) and isinstance(cm, str):
                combo[(am, cm)] += 1
        if not combo:
            continue
        items = sorted(combo.items(), key=lambda kv: kv[1], reverse=True)
        pretty = ", ".join([f"{a}->{c}:{n}" for (a, c), n in items])
        print(f"  {name}: {pretty}")
    print("")

    # Distributions for common metadata keys
    if show_distributions:
        keys = collect_common_keys(by_model)
        if keys:
            print("Distributions (per model):")
            for name, trials in per_model_trials:
                print(f"\n  --- {name} ---")
                for k in keys:
                    if k in {"A_mode", "C_mode"}:
                        continue
                    ds = distribution_summary(trials, k)
                    if ds["type"] == "numeric":
                        mean = ds["mean"]
                        q10, q50, q90 = ds["q10_q50_q90"] or (None, None, None)
                        print(f"  {k:16s}  n={ds['count']:4d}  mean={mean:.3f}  q10/q50/q90={q10:.3f}/{q50:.3f}/{q90:.3f}")
                    elif ds["type"] == "categorical":
                        top = ", ".join([f"{v}:{c}" for v, c in ds["top"]])
                        print(f"  {k:16s}  n={ds['count']:4d}  top={top}")
        print("\n")

    # Error examples
    if show_errors:
        print("Failure examples (first few, per model):")
        for name, trials in per_model_trials:
            print(f"\n  --- {name} ---")
            print_error_examples(trials, max_examples=8)
        print("")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="Path to a run dir, e.g. analogies/experiments/runs/<run_id>")
    ap.add_argument("--no-dists", action="store_true", help="Skip distribution printing")
    ap.add_argument("--no-errors", action="store_true", help="Skip error examples")
    args = ap.parse_args()

    report(args.run_dir, show_distributions=not args.no_dists, show_errors=not args.no_errors)

if __name__ == "__main__":
    main()