from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Sibling import of make_latex_tables
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import make_latex_tables as mlt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a single LaTeX table combining main accuracy and agreement blocks."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="Run directory containing trials.ndjson (defaults to runs/gold_run).",
    )
    return parser.parse_args()


def _default_run_dir() -> Path:
    return _HERE / "runs" / "gold_run"


def _load_human_mean_agreement() -> float:
    path = _HERE / "human_identity_responses" / "identity_human_agreement_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing human summary JSON at {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    sets = payload.get("sets", [])
    if not sets:
        raise ValueError(f"No human sets in {path}")
    vals = [float(s.get("mean_majority_agreement_pct", 0.0)) / 100.0 for s in sets]
    return sum(vals) / len(vals)


def build_combined_table(trials: List[Dict[str, Any]]) -> str:
    main_rows = mlt.main_result_rows(trials)
    agree_rows = mlt.agreement_result_rows(trials)
    agree_by_model: Dict[str, List[str]] = {mid: cells for mid, cells in agree_rows}
    main_ids = {mid for mid, _ in main_rows}
    agree_ids = set(agree_by_model.keys())
    if main_ids != agree_ids:
        missing_main = agree_ids - main_ids
        missing_agree = main_ids - agree_ids
        raise ValueError(
            "Model sets differ between main and agreement metrics: "
            f"only in agreement {missing_main!r}, only in main {missing_agree!r}"
        )

    body_rows: List[List[str]] = []
    for model_id, main_cells in main_rows:
        display, acc_cell = main_cells[0], main_cells[1]
        agree_cells = agree_by_model[model_id]
        if agree_cells[0] != display:
            raise ValueError(
                f"Display name mismatch for {model_id!r}: {display!r} vs {agree_cells[0]!r}"
            )
        # keep only agreement probabilities (drop entropy columns)
        body_rows.append([display, acc_cell, agree_cells[1], agree_cells[3], agree_cells[5]])

    ncols = 5
    cols = "l" + "r" * (ncols - 1)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\renewcommand{\\arraystretch}{1.08}",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        "\\multicolumn{2}{@{}c@{}}{\\textbf{Main accuracy}} & "
        "\\multicolumn{3}{c@{}}{\\textbf{Agreement statistics}} \\\\",
        "\\cmidrule(r){1-2} \\cmidrule(l){3-5}",
        "Model & Accuracy & $P(agree)$ & "
        "$P(agree\\mid any\\ wrong)$ & "
        "$P(agree\\mid all\\ wrong)$ \\\\",
        "\\midrule",
    ]
    for row in body_rows:
        lines.append(" & ".join(row) + " \\\\")
    human_mean_agree = _load_human_mean_agreement()
    human_cell = mlt._fmt_est_se(human_mean_agree, 0.0)
    lines.append("\\midrule")
    lines.append(
        " & ".join(
            [
                "\\textbf{Human}",
                f"\\textbf{{{human_cell}}}",
                f"\\textbf{{{human_cell}}}",
                "\\textbf{--}",
                "\\textbf{--}",
            ]
        )
        + " \\\\"
    )
    cap = (
        "Identity-copy accuracy and three-iteration agreement statistics per model. "
        "The final bold Human row is aggregated across all 5 human sets. "
        "Standard errors are in parentheses."
    )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{{cap}}}",
            "\\label{tab:identity_combined}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else _default_run_dir()
    trials_path = run_dir / "trials.ndjson"
    if not trials_path.exists():
        raise FileNotFoundError(f"Missing trials.ndjson at {trials_path}")

    trials = mlt.load_trials(trials_path)
    out = run_dir / "latex_table_combined.tex"
    out.write_text(build_combined_table(trials) + "\n", encoding="utf-8")
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
