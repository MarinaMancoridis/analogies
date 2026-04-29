from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import run_relation_triples as rrt


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Create a repaired copy of a relations run by retrying only rows with "
            "error != null or empty D."
        )
    )
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        required=True,
        help="Path to the original run directory (will not be modified).",
    )
    parser.add_argument(
        "--out-run-dir",
        type=Path,
        default=None,
        help="Optional explicit output run directory. Defaults to repair-<source>-<timestamp>.",
    )
    return parser.parse_args()


def _load_ndjson(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _is_retry_target(row: Dict[str, Any]) -> bool:
    has_error = bool(row.get("error"))
    empty_d = not str(row.get("D") or "").strip()
    return has_error or empty_d


def _print_progress(done: int, total: int) -> None:
    width = 30
    total = max(total, 1)
    frac = min(1.0, done / total)
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(f"\rRetrying rows [{bar}] {done}/{total} ({frac * 100:5.1f}%)")
    sys.stdout.flush()
    if done >= total:
        sys.stdout.write("\n")


def _retry_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    model = str(out["model"])
    relation_key = str(out["relation_key"])
    a = str(out["A"])
    b = str(out["B"])
    c = str(out["C"])
    prompt_type = str(out["prompt_type"])

    module = rrt._load_relation_module(relation_key)

    error = None
    d = ""
    raw_d = ""
    prompt_d = ""
    judge: Dict[str, Any] = {
        "judge_prompt": "",
        "judge_raws": [],
        "judge_labels": [],
        "judge_counts": {},
        "judge_majority_label": None,
        "judge_majority_correct": False,
        "judge_n": 0,
        "judge_error": "",
    }

    try:
        prompt_d = module.PROMPT_VARIATIONS[prompt_type](a, b, c)
        d, raw_d = module._ask_one(model, prompt_d)
        judge = rrt._run_relation_judge(module, model, a, b, c, d)
    except Exception as e:  # keep row with failure metadata if retry still fails
        error = str(e)

    out["timestamp_utc"] = rrt._utc_now()
    out["D"] = d
    out["D_metrics"] = rrt._word_metrics_or_none(module, d)
    out["prompt_complete"] = prompt_d
    out["raw_response_D"] = raw_d
    out["grade_prompt"] = judge["judge_prompt"]
    out["grade_raw"] = judge["judge_raws"][-1] if judge["judge_raws"] else ""
    out["grade_correct"] = judge["judge_majority_correct"]
    out["judge_raws"] = judge["judge_raws"]
    out["judge_labels"] = judge["judge_labels"]
    out["judge_counts"] = judge["judge_counts"]
    out["judge_majority_label"] = judge["judge_majority_label"]
    out["judge_majority_correct"] = judge["judge_majority_correct"]
    out["judge_n"] = judge["judge_n"]
    out["judge_error"] = judge["judge_error"]
    out["error"] = error
    return out


def _compute_worker_done(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"n": 0, "b_nonempty": 0, "d_nonempty": 0, "errors": 0})
    for r in rows:
        m = str(r.get("model", ""))
        s = stats[m]
        s["n"] += 1
        s["b_nonempty"] += int(bool(str(r.get("B") or "").strip()))
        s["d_nonempty"] += int(bool(str(r.get("D") or "").strip()))
        s["errors"] += int(bool(r.get("error")))
    for m, s in stats.items():
        s["model"] = m
        s["d_nonempty_rate"] = (s["d_nonempty"] / s["n"]) if s["n"] else 0.0
    return stats


def main() -> None:
    args = parse_args()
    source_run = args.source_run_dir.expanduser().resolve()
    if not source_run.exists():
        raise FileNotFoundError(source_run)

    source_trials = source_run / "trials.ndjson"
    source_manifest = source_run / "run_manifest.json"
    if not source_trials.exists() or not source_manifest.exists():
        raise FileNotFoundError("Source run is missing trials.ndjson or run_manifest.json")

    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    rows = _load_ndjson(source_trials)
    expected = int(manifest.get("expected_generations", 0))
    if expected and len(rows) != expected:
        raise ValueError(f"Source run has {len(rows)} rows, expected {expected}")

    retry_indices = [i for i, r in enumerate(rows) if _is_retry_target(r)]
    print(f"Source rows: {len(rows)}")
    print(f"Rows to retry: {len(retry_indices)}")

    ts = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out_run = (
        args.out_run_dir.expanduser().resolve()
        if args.out_run_dir
        else source_run.parent / f"repair-{source_run.name}-{ts}"
    )
    if out_run.exists():
        raise FileExistsError(f"Output run already exists: {out_run}")
    out_run.mkdir(parents=True, exist_ok=False)
    (out_run / "worker_done").mkdir(parents=True, exist_ok=True)

    repaired_rows = list(rows)
    _print_progress(0, len(retry_indices))
    for n, idx in enumerate(retry_indices, start=1):
        repaired_rows[idx] = _retry_row(repaired_rows[idx])
        _print_progress(n, len(retry_indices))

    out_manifest = dict(manifest)
    out_manifest["created_at_utc"] = rrt._utc_now()
    out_manifest["repair_of_run"] = str(source_run)
    out_manifest["repair_reason"] = "retry rows with error or empty D"
    out_manifest["repair_retried_rows"] = len(retry_indices)
    out_manifest["repair_expected_rows"] = len(rows)
    (out_run / "run_manifest.json").write_text(
        json.dumps(out_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with (out_run / "trials.ndjson").open("w", encoding="utf-8") as f:
        for r in repaired_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    worker_stats = _compute_worker_done(repaired_rows)
    for model, stats in worker_stats.items():
        slug = rrt._slugify_model(model)
        (out_run / "worker_done" / f"{slug}.json").write_text(
            json.dumps(stats, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    rrt._aggregate_summary(str(out_run))
    print(f"Wrote repaired run: {out_run}")


if __name__ == "__main__":
    main()
