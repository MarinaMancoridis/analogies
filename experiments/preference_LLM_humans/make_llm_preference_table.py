from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

PRETTY_MODEL_NAMES: Dict[str, str] = {
    "meta-llama/llama-3.3-70b-instruct": "Llama 3.3 70B Instruct",
    "gpt-4o": "GPT-4o",
    "openai/gpt-4.1": "GPT-4.1",
    "openai/gpt-5.4-mini": "GPT-5.4 Mini",
    "google/gemini-3-flash-preview": "Gemini 3 Flash Preview",
    "google/gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash-Lite Preview",
    "anthropic/claude-sonnet-4.5": "Claude Sonnet 4.5",
    "anthropic/claude-opus-4.6": "Claude Opus 4.6",
    "deepseek-ai/DeepSeek-V3": "DeepSeek V3",
    "deepseek/deepseek-v3.2": "DeepSeek V3.2",
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Build LaTeX table for LLM pairwise preference results."
    )
    parser.add_argument(
        "--results-ndjson",
        type=Path,
        default=here / "pairwise_llm_judgments_colon.ndjson",
        help=(
            "NDJSON from run_pairwise_preferences.py "
            "(default: colon-filtered judgments)."
        ),
    )
    parser.add_argument(
        "--out-tex",
        type=Path,
        default=here / "llm_preference_table_colon.tex",
        help="Output LaTeX table path (default avoids overwriting llm_preference_table.tex).",
    )
    parser.add_argument(
        "--human-analysis-json",
        type=Path,
        default=here / "human_preference_analysis.json",
        help=(
            "Output from analyze_human_preference_responses.py; used for the Human "
            "evaluator row (overall.n_prefer_human_completion / overall.n_judgments)."
        ),
    )
    return parser.parse_args()


def _read_latest_run_summary(path: Path) -> Dict[str, Any]:
    latest: Dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("record_type") == "run_summary":
                latest = rec
    if latest is None:
        raise ValueError(f"No run_summary record found in: {path}")
    return latest


def _binomial_se(successes: int, total: int) -> float:
    if total <= 0:
        return 0.0
    p = successes / total
    return math.sqrt(p * (1.0 - p) / total)


def _fmt_prop(x: float) -> str:
    return f"{x:.2g}"


def _pretty_model_name(model: str) -> str:
    return PRETTY_MODEL_NAMES.get(model, model)


def _load_human_overall_counts(path: Path) -> tuple[int, int]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    o = data["overall"]
    return int(o["n_prefer_human_completion"]), int(o["n_judgments"])


def build_table(
    run_summary: Dict[str, Any],
    human_prefer: int,
    human_n_judgments: int,
) -> str:
    summary_by_model: Dict[str, Dict[str, Any]] = run_summary["summary_by_model"]
    judge_models: List[str] = run_summary.get("judge_models", list(summary_by_model.keys()))

    lines: List[str] = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lr}")
    lines.append("\\toprule")
    lines.append("Evaluator & Human-chosen proportion \\\\")
    lines.append("\\midrule")

    for model in judge_models:
        s = summary_by_model[model]
        n_valid = int(s["n_valid"])
        n_invalid = int(s["n_invalid"])
        human = int(s["human_chosen"])
        p = (human / n_valid) if n_valid else 0.0
        se = _binomial_se(human, n_valid)
        display_name = _pretty_model_name(model)
        lines.append(f"{display_name} & {_fmt_prop(p)} ({_fmt_prop(se)}) \\\\")

    lines.append("\\midrule")
    hp = (human_prefer / human_n_judgments) if human_n_judgments else 0.0
    hse = _binomial_se(human_prefer, human_n_judgments)
    lines.append(
        f"\\textbf{{Human}} & \\textbf{{{_fmt_prop(hp)} ({_fmt_prop(hse)})}} \\\\"
    )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(
        "\\caption{Pairwise preference results by evaluator, restricted to comparisons where the model completion used the \\texttt{colon} prompt type. "
        "Values are the proportion of judgments where the evaluator preferred the human-sourced completion over the model completion, with binomial standard errors in parentheses. "
        "The \\textbf{Human} row aggregates eligible survey responses (all item sets) from \\texttt{human\\_preference\\_analysis.json} (same coding as the analysis script).}"
    )
    lines.append("\\label{tab:llm_preference_table_colon}")
    lines.append("\\end{table}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_summary = _read_latest_run_summary(args.results_ndjson)
    h_pref, h_n = _load_human_overall_counts(args.human_analysis_json)
    tex = build_table(run_summary, h_pref, h_n)
    args.out_tex.write_text(tex, encoding="utf-8")
    print(f"Wrote: {args.out_tex}")


if __name__ == "__main__":
    main()
