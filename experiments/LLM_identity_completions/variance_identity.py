#!/usr/bin/env python3
"""
Agreement / variance across repeated identity completions (same prompt, 3 iterations).

Reads ``runs/gold_run/trials.ndjson`` (or ``--trials``) and prints a LaTeX table with
per-model statistics over triples (each triple = 3 generations for that model).

Usage:
  python variance_identity.py
  python variance_identity.py --trials runs/gold_run/trials.ndjson --out-tex agreement_table.tex
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

HERE = Path(__file__).resolve().parent

PRETTY_MODEL_NAMES: Dict[str, str] = {
    "meta-llama/llama-3.3-70b-instruct": "Llama 3.3 70B",
    "gpt-4o": "GPT-4o",
    "openai/gpt-4.1": "GPT-4.1",
    "openai/gpt-5.4-mini": "GPT-5.4 Mini",
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    "google/gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash-Lite",
    "anthropic/claude-sonnet-4.5": "Claude Sonnet 4.5",
    "anthropic/claude-opus-4.6": "Claude Opus 4.6",
    "deepseek-ai/DeepSeek-V3": "DeepSeek V3",
    "deepseek/deepseek-v3.2": "DeepSeek V3.2",
}

MODEL_ORDER = list(PRETTY_MODEL_NAMES.keys())


def _binomial_se(k: int, n: int) -> float:
    if n <= 0:
        return 0.0
    p = k / n
    return math.sqrt(p * (1.0 - p) / n)


def _tex_escape(s: str) -> str:
    out: List[str] = []
    for ch in s:
        if ch in ("&", "%", "#", "_"):
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out)


def _pairwise_agreement_fraction(answers: List[str]) -> float:
    """Three answers: fraction of the 3 unordered pairs that match."""
    if len(answers) != 3:
        raise ValueError("expected 3 answers")
    same = sum(
        1 for i in range(3) for j in range(i + 1, 3) if answers[i] == answers[j]
    )
    return same / 3.0


def _load_trials(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _group_by_model_triple(
    rows: List[Dict[str, Any]],
) -> Dict[Tuple[str, int], List[Dict[str, Any]]]:
    g: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("experiment_name") != "LLM_identity_completions":
            continue
        model = str(r["model"])
        tid = int(r["triple_id"])
        g[(model, tid)].append(r)
    for key in g:
        g[key].sort(key=lambda x: int(x["iteration"]))
    return g


def _compute_model_stats(
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]], model: str
) -> Dict[str, Any]:
    triple_keys = [k for k in grouped if k[0] == model]
    triple_keys.sort(key=lambda x: x[1])

    n_unanimous = 0
    n_majority2 = 0
    n_all_three_distinct = 0
    n_all_identity = 0
    pairwise_fracs: List[float] = []
    # Error consistency: among triples where NOT all three draws are correct,
    # how often all three parsed tokens still agree (same wrong—or rare mixed—output).
    n_not_all_correct = 0
    n_agree_given_not_all_correct = 0

    bad_n = 0
    for _m, tid in triple_keys:
        trials = grouped[(_m, tid)]
        if len(trials) != 3:
            bad_n += 1
            continue
        parsed = [str(t.get("parsed_answer") or "").strip().lower() for t in trials]
        ident = [bool(t.get("is_identity")) for t in trials]

        ctr = Counter(parsed)
        mx = max(ctr.values())
        three_same = mx == 3
        if three_same:
            n_unanimous += 1
        if mx >= 2:
            n_majority2 += 1
        if len(ctr) == 3:
            n_all_three_distinct += 1
        all_corr = all(ident)
        if all_corr:
            n_all_identity += 1
        if not all_corr:
            n_not_all_correct += 1
            if three_same:
                n_agree_given_not_all_correct += 1
        pairwise_fracs.append(_pairwise_agreement_fraction(parsed))

    n = len(triple_keys) - bad_n
    if bad_n:
        pass  # caller can warn

    mean_pair = statistics.mean(pairwise_fracs) if pairwise_fracs else 0.0
    # SE of mean of triple-level fractions (treat triples as i.i.d.)
    if len(pairwise_fracs) > 1:
        se_pair = statistics.stdev(pairwise_fracs) / math.sqrt(len(pairwise_fracs))
    else:
        se_pair = 0.0

    a_rate = n_all_identity / n if n else 0.0
    gap_consistency_minus_correct = mean_pair - a_rate

    return {
        "n_triples": n,
        "n_bad_group_size": bad_n,
        "unanimous_token": (n_unanimous, n),
        "majority_ge2": (n_majority2, n),
        "all_three_distinct": (n_all_three_distinct, n),
        "all_identity": (n_all_identity, n),
        "mean_pairwise_agree": (mean_pair, se_pair),
        "gap_pairwise_minus_all_correct": gap_consistency_minus_correct,
        "error_consistency": (
            n_agree_given_not_all_correct,
            n_not_all_correct,
        ),
    }


def build_latex_table(
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]],
    models: List[str],
) -> str:
    row_list: List[Tuple[str, Dict[str, Any]]] = []
    for model in models:
        if not any(k[0] == model for k in grouped):
            continue
        row_list.append((model, _compute_model_stats(grouped, model)))

    # Worst consistency excess first (high gap = bad).
    row_list.sort(key=lambda x: x[1]["gap_pairwise_minus_all_correct"], reverse=True)

    top_k = 3
    bold_gap_models = {row_list[i][0] for i in range(min(top_k, len(row_list)))}

    ec_ranked: List[Tuple[str, float]] = []
    for model, st in row_list:
        ec_k, ec_n = st["error_consistency"]
        if ec_n > 0:
            ec_ranked.append((model, ec_k / ec_n))
    ec_ranked.sort(key=lambda x: x[1], reverse=True)
    bold_rw_models = {ec_ranked[i][0] for i in range(min(top_k, len(ec_ranked)))}

    lines: List[str] = [
        "% Requires: \\usepackage{booktabs}",
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}l ccc | ccc@{}}",
        "\\toprule",
        "\\textbf{Model} & "
        "\\multicolumn{3}{c}{\\textbf{(A) Consistency}} & "
        "\\multicolumn{3}{c}{\\textbf{(B) Correctness \\& contrast}} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
        "& \\textbf{3 same} & \\textbf{$\\geq$2} & \\textbf{pairw.} "
        "& \\textbf{all corr.} & "
        "\\textbf{\\shortstack{consistency gap\\\\$\\uparrow$\\textsuperscript{\\textit{a}}}} & "
        "\\textbf{\\shortstack{repeat when wrong\\\\$\\uparrow$\\textsuperscript{\\textit{b}}}} \\\\",
        "\\midrule",
    ]

    for model, st in row_list:
        pretty = _tex_escape(PRETTY_MODEL_NAMES.get(model, model))
        u_k, u_n = st["unanimous_token"]
        m_k, m_n = st["majority_ge2"]
        a_k, a_n = st["all_identity"]
        mp, mpse = st["mean_pairwise_agree"]
        gap = st["gap_pairwise_minus_all_correct"]
        ec_k, ec_n = st["error_consistency"]

        cell_u = f"{u_k / u_n:.2f} ({_binomial_se(u_k, u_n):.3f})" if u_n else "---"
        cell_m = f"{m_k / m_n:.2f} ({_binomial_se(m_k, m_n):.3f})" if m_n else "---"
        cell_p = f"{mp:.2f} ({mpse:.3f})" if u_n else "---"
        cell_a = f"{a_k / a_n:.2f} ({_binomial_se(a_k, a_n):.3f})" if a_n else "---"

        gap_body = f"{gap:+.2f}"
        if u_n and model in bold_gap_models:
            gap_body = f"\\textbf{{{gap_body}}}"
        cell_gap = gap_body if u_n else "---"

        if ec_n > 0:
            ec_body = f"{ec_k / ec_n:.2f} ({_binomial_se(ec_k, ec_n):.3f})"
            if model in bold_rw_models:
                ec_body = f"\\textbf{{{ec_body}}}"
            cell_ec = ec_body
        else:
            cell_ec = "---"

        lines.append(
            f"\\scriptsize {pretty} & \\scriptsize {cell_u} & \\scriptsize {cell_m} "
            f"& \\scriptsize {cell_p} & \\scriptsize {cell_a} & \\scriptsize {cell_gap} "
            f"& \\scriptsize {cell_ec} \\\\"
        )

    cap = (
        "\\caption{Models exhibit excess internal consistency relative to correctness on a trivial identity "
        "task (\\texttt{gold\\_run}: $n$ triples per model, three i.i.d.\\ draws per prompt). "
        "The \\textbf{consistency gap} (pairwise agreement minus joint correctness) highlights regimes where "
        "models agree with themselves more than with the task; "
        "\\textsuperscript{\\textit{a}}\\,\\textbf{higher} = \\textbf{more consistent than correct}. "
        "\\textbf{Repeat when wrong} measures how often models produce the same (typically incorrect) answer "
        "across all draws when at least one draw is wrong---\\textsuperscript{\\textit{b}}\\,a directly "
        "behavioral readout (\\textbf{higher} = more stubborn repetition). "
        "\\textbf{(A)} \\emph{3 same} / \\emph{$\\geq$2} / \\emph{pairw.}; "
        "\\textbf{(B)} \\emph{all corr.}\\ = all \\texttt{is\\_identity}. "
        "\\textbf{Bold} = top three on each contrast column. Binomial SEs in parentheses; "
        "``---'' if no imperfect triples.}"
    )

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            cap,
            "\\label{tab:identity_repeat_agreement}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Identity run agreement / variance table.")
    p.add_argument(
        "--trials",
        type=Path,
        default=HERE / "runs" / "gold_run" / "trials.ndjson",
        help="Path to trials.ndjson",
    )
    p.add_argument(
        "--out-tex",
        type=Path,
        default=None,
        help="If set, write LaTeX here; otherwise only print to stdout.",
    )
    args = p.parse_args()

    if not args.trials.is_file():
        raise SystemExit(f"Trials file not found: {args.trials}")

    rows = _load_trials(args.trials)
    grouped = _group_by_model_triple(rows)
    models_in_data = sorted({k[0] for k in grouped}, key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 999)
    tex = build_latex_table(grouped, models_in_data)

    print("=== Identity repeat agreement (gold_run), sorted by consistency gap (high first) ===", flush=True)
    console_rows: List[Tuple[str, Dict[str, Any]]] = [
        (m, _compute_model_stats(grouped, m))
        for m in models_in_data
        if any(k[0] == m for k in grouped)
    ]
    console_rows.sort(key=lambda x: x[1]["gap_pairwise_minus_all_correct"], reverse=True)
    for model, st in console_rows:
        u_k, u_n = st["unanimous_token"]
        d_k, d_n = st["all_three_distinct"]
        ec_k, ec_n = st["error_consistency"]
        ec_s = f"{ec_k}/{ec_n}={ec_k/ec_n:.3f}" if ec_n else "n/a"
        print(
            f"  {model}: n_triples={st['n_triples']} | "
            f"3-same={u_k}/{u_n} | "
            f"3-distinct={d_k}/{d_n} | "
            f"all_identity={st['all_identity'][0]}/{st['all_identity'][1]} | "
            f"cons. gap={st['gap_pairwise_minus_all_correct']:+.3f} | "
            f"repeat-when-wrong={ec_s}",
            flush=True,
        )
    print(flush=True)
    print(tex)
    if args.out_tex is not None:
        args.out_tex.parent.mkdir(parents=True, exist_ok=True)
        args.out_tex.write_text(tex + "\n", encoding="utf-8")
        print(f"\nWrote {args.out_tex}", flush=True)


if __name__ == "__main__":
    main()
