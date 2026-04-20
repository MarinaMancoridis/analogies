from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Tuple

# Repo layout: .../analogies/{experiments/.../this_file.py, constants.py, ...}
# so `import analogies` needs the parent of the repo folder on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT.parent))

from analogies.constants import models_to_short_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables for identity experiment results."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="Run directory containing trials.ndjson (defaults to runs/gold_run).",
    )
    return parser.parse_args()


def _default_run_dir() -> Path:
    here = Path(__file__).resolve().parent
    return here / "runs" / "gold_run"


def _round_sig(x: float, sig: int) -> float:
    """Round non-negative `x` to `sig` significant figures."""
    if x == 0 or not math.isfinite(x):
        return 0.0
    m = math.floor(math.log10(abs(x)))
    factor = 10 ** (sig - 1 - m)
    return round(x * factor) / factor


def _format_sig_str(x: float, sig: int) -> str:
    """Format `x` with exactly `sig` significant digits (decimal notation)."""
    if not math.isfinite(x):
        return "nan"
    y = _round_sig(x, sig)
    if y == 0:
        return "0"
    ay = abs(y)
    m = int(math.floor(math.log10(ay)))
    decimals = max(0, (sig - 1) - m)
    return f"{y:.{decimals}f}"


def _binom_se(p: float, n: int) -> float:
    if n <= 0:
        return 0.0
    return math.sqrt(max(0.0, p * (1.0 - p) / n))


def _mean_sem(vals: List[float]) -> Tuple[float, float]:
    """Sample mean and standard error of the mean (single draw per triple)."""
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    m = mean(vals)
    if n < 2:
        return m, 0.0
    return m, stdev(vals) / math.sqrt(n)


def _fmt_est_se(val: float, se: float) -> str:
    return f"{_format_sig_str(val, 2)} ({_format_sig_str(se, 1)})"


def _main_table_sort_key(model_id: str, display: str) -> Tuple[int, int, str]:
    """Order rows by provider, then oldest->newest model within provider."""
    mid = model_id.lower()
    if mid.startswith("anthropic/"):
        rank = 0
    elif mid.startswith("google/") or mid.startswith("gemini-"):
        rank = 1
    elif (
        mid.startswith("openai/")
        or mid.startswith("gpt-")
        or mid.startswith("o1")
        or mid.startswith("o3")
    ):
        rank = 2
    elif mid.startswith("deepseek"):
        rank = 3
    elif mid.startswith("meta-llama/"):
        rank = 4
    else:
        rank = 99

    # Explicit within-provider chronology for known models used in this experiment.
    within_provider_rank = {
        "anthropic/claude-sonnet-4.5": 0,
        "anthropic/claude-opus-4.6": 1,
        "google/gemini-3-flash-preview": 0,
        "google/gemini-3.1-flash-lite-preview": 1,
        "gpt-4o": 0,
        "openai/gpt-4.1": 1,
        "openai/gpt-5.4-mini": 2,
        "deepseek-ai/deepseek-v3": 0,
        "deepseek/deepseek-v3.2": 1,
        "meta-llama/llama-3.3-70b-instruct": 0,
    }.get(mid, 999)
    return (rank, within_provider_rank, display.lower())


def _entropy(vals: List[str]) -> float:
    n = len(vals)
    if n == 0:
        return 0.0
    counts = Counter(vals)
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log2(p)
    return h


def _to_latex_table(
    header: List[str],
    rows: List[List[str]],
    caption: str,
    label: str,
    *,
    booktabs: bool = True,
    caption_below: bool = False,
) -> str:
    cols = "l" + "r" * (len(header) - 1)
    top = "\\toprule" if booktabs else "\\hline"
    mid = "\\midrule" if booktabs else "\\hline"
    bottom = "\\bottomrule" if booktabs else "\\hline"
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\renewcommand{\\arraystretch}{1.08}",
    ]
    if not caption_below:
        lines.extend([f"\\caption{{{caption}}}", f"\\label{{{label}}}"])
    lines.extend(
        [
            f"\\begin{{tabular}}{{{cols}}}",
            top,
            " & ".join(header) + " \\\\",
            mid,
        ]
    )
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.extend([bottom, "\\end{tabular}"])
    if caption_below:
        lines.extend([f"\\caption{{{caption}}}", f"\\label{{{label}}}"])
    lines.append("\\end{table}")
    return "\n".join(lines)


def load_trials(trials_path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with trials_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def main_result_rows(trials: List[Dict[str, Any]]) -> List[Tuple[str, List[str]]]:
    """Sorted rows: each item is (raw_model_id, [display_name, accuracy_cell])."""
    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for t in trials:
        by_model[str(t.get("model", ""))].append(t)

    sortable_rows: List[Tuple[Tuple[int, str], str, List[str]]] = []
    for model in by_model.keys():
        model_trials = by_model[model]
        n = len(model_trials)
        hits = sum(1 for t in model_trials if t.get("is_identity"))
        acc = hits / n if n else 0.0
        se = _binom_se(acc, n)
        display = models_to_short_name.get(model, model)
        sortable_rows.append(
            (
                _main_table_sort_key(model, display),
                model,
                [
                    display,
                    _fmt_est_se(acc, se),
                ],
            )
        )
    return [(m, cells) for _k, m, cells in sorted(sortable_rows, key=lambda x: x[0])]


def agreement_result_rows(trials: List[Dict[str, Any]]) -> List[Tuple[str, List[str]]]:
    """Sorted rows: each item is (raw_model_id, [display, p_agree, H, ...])."""
    groups: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for t in trials:
        model = str(t.get("model", ""))
        triple_id = int(t.get("triple_id", -1))
        groups[(model, triple_id)].append(t)

    per_model_groups: Dict[str, List[List[Dict[str, Any]]]] = defaultdict(list)
    for (model, _triple_id), group in groups.items():
        per_model_groups[model].append(group)

    row_entries: List[Tuple[Tuple[int, str], str, List[str]]] = []
    for model, model_groups in per_model_groups.items():
        all_agree_count = 0
        any_wrong_count = 0
        any_wrong_and_agree_count = 0
        all_wrong_count = 0
        all_wrong_and_agree_count = 0
        entropies: List[float] = []
        wrong_entropies: List[float] = []

        for g in model_groups:
            g_sorted = sorted(g, key=lambda x: int(x.get("iteration", 0)))
            answers = [str(x.get("parsed_answer", "")) for x in g_sorted]
            correctness = [bool(x.get("is_identity")) for x in g_sorted]

            agree = len(set(answers)) == 1
            h = _entropy(answers)
            entropies.append(h)

            if agree:
                all_agree_count += 1

            if not all(correctness):
                any_wrong_count += 1
                wrong_entropies.append(h)
                if agree:
                    any_wrong_and_agree_count += 1

            if not any(correctness):
                all_wrong_count += 1
                if agree:
                    all_wrong_and_agree_count += 1

        n_groups = len(model_groups)
        p_all_agree = all_agree_count / n_groups if n_groups else 0.0
        se_all_agree = _binom_se(p_all_agree, n_groups)
        p_agree_given_any_wrong = (
            any_wrong_and_agree_count / any_wrong_count if any_wrong_count else 0.0
        )
        se_agree_any_wrong = _binom_se(p_agree_given_any_wrong, any_wrong_count)
        p_agree_given_all_wrong = (
            all_wrong_and_agree_count / all_wrong_count if all_wrong_count else 0.0
        )
        se_agree_all_wrong = _binom_se(p_agree_given_all_wrong, all_wrong_count)
        mean_h, se_h = _mean_sem(entropies)
        mean_h_wrong, se_h_wrong = _mean_sem(wrong_entropies)

        display = models_to_short_name.get(model, model)
        row_entries.append(
            (
                _main_table_sort_key(model, display),
                model,
                [
                    display,
                    _fmt_est_se(p_all_agree, se_all_agree),
                    _fmt_est_se(mean_h, se_h),
                    _fmt_est_se(p_agree_given_any_wrong, se_agree_any_wrong),
                    _fmt_est_se(mean_h_wrong, se_h_wrong),
                    _fmt_est_se(p_agree_given_all_wrong, se_agree_all_wrong),
                ],
            )
        )

    return [(m, cells) for _k, m, cells in sorted(row_entries, key=lambda x: x[0])]


def build_main_results_table(trials: List[Dict[str, Any]]) -> str:
    rows = [cells for _mid, cells in main_result_rows(trials)]

    return _to_latex_table(
        header=["Model", "Accuracy"],
        rows=rows,
        caption=(
            "Main identity-copy results on the gold run. "
            "Standard errors are in parentheses."
        ),
        label="tab:identity_main_results",
        booktabs=True,
        caption_below=True,
    )


def build_agreement_entropy_table(trials: List[Dict[str, Any]]) -> str:
    rows = [cells for _mid, cells in agreement_result_rows(trials)]

    return _to_latex_table(
        header=[
            "Model",
            "$P(agree)$",
            "$H$",
            "$P(agree\\mid any\\ wrong)$",
            "$H\\mid any\\ wrong$",
            "$P(agree\\mid all\\ wrong)$",
        ],
        rows=rows,
        caption=(
            "Three-iteration agreement and entropy per model. "
            "$H$ is answer entropy (bits) across the three repeated runs per triple. "
            "Standard errors are in parentheses."
        ),
        label="tab:identity_agreement_entropy",
        booktabs=True,
    )


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else _default_run_dir()
    trials_path = run_dir / "trials.ndjson"
    if not trials_path.exists():
        raise FileNotFoundError(f"Missing trials.ndjson at {trials_path}")

    trials = load_trials(trials_path)
    main_table = build_main_results_table(trials)
    agree_table = build_agreement_entropy_table(trials)

    out_main = run_dir / "latex_table_main_results.tex"
    out_agree = run_dir / "latex_table_agreement_entropy.tex"
    out_main.write_text(main_table + "\n", encoding="utf-8")
    out_agree.write_text(agree_table + "\n", encoding="utf-8")

    print(f"Wrote: {out_main}")
    print(f"Wrote: {out_agree}")


if __name__ == "__main__":
    main()
