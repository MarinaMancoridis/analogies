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

MODEL_DISPLAY_ORDER: List[str] = list(PRETTY_MODEL_NAMES.keys())

# Short column headers (full names still documented in caption).
SHORT_RELATION_HEADER: Dict[str, str] = {
    "attribute": "Attr.",
    "case_relations": "Case",
    "cause_purpose": "C--P",
    "class_inclusion": "C--Inc.",
    "contrast": "Contr.",
    "non_attribute": "Non-att.",
    "part_whole": "P--W",
    "reference": "Ref.",
    "similar": "Sim.",
    "space_time": "S--T",
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description=(
            "Build a LaTeX table from llm_validate_relation_following.json "
            "(compact wide layout: models $\\times$ relations)."
        )
    )
    p.add_argument(
        "--in-json",
        type=Path,
        default=here / "llm_validate_relation_following.json",
        help="Output of LLM_validate_relation_following.py.",
    )
    p.add_argument(
        "--out-tex",
        type=Path,
        default=here / "relation_following_validation_table.tex",
        help="Output .tex path.",
    )
    return p.parse_args()


def _binomial_se(successes: int, total: int) -> float:
    if total <= 0:
        return 0.0
    p = successes / total
    return math.sqrt(p * (1.0 - p) / total)


def _fmt_prop_se(p: float, se: float) -> str:
    return f"{p:.2f} ({se:.3f})"


def _tex_escape(s: str) -> str:
    out: List[str] = []
    for ch in s:
        if ch == "\\":
            out.append("\\textbackslash{}")
        elif ch == "&":
            out.append("\\&")
        elif ch == "%":
            out.append("\\%")
        elif ch == "$":
            out.append("\\$")
        elif ch == "#":
            out.append("\\#")
        elif ch == "_":
            out.append("\\_")
        elif ch == "{":
            out.append("\\{")
        elif ch == "}":
            out.append("\\}")
        elif ch == "~":
            out.append("\\textasciitilde{}")
        elif ch == "^":
            out.append("\\textasciicircum{}")
        else:
            out.append(ch)
    return "".join(out)


def _sort_models(models: List[str]) -> List[str]:
    rank = {m: i for i, m in enumerate(MODEL_DISPLAY_ORDER)}
    return sorted(models, key=lambda m: (rank.get(m, 10_000), m))


def _pretty_model(model: str) -> str:
    return PRETTY_MODEL_NAMES.get(model, model)


def _col_header_tex(relation_key: str, relation_name: str) -> str:
    short = SHORT_RELATION_HEADER.get(relation_key)
    if short is None:
        short = _tex_escape(relation_name[:8] + ("…" if len(relation_name) > 8 else ""))
    else:
        short = _tex_escape(short)
    return short


def _caption_column_legend(rel_keys: List[str], by_relation: Dict[str, Any]) -> str:
    parts: List[str] = []
    for rk in rel_keys:
        abbrev = SHORT_RELATION_HEADER.get(rk)
        full = str(by_relation[rk]["relation_name"])
        if abbrev:
            parts.append(
                "\\textit{%s}\\,=\\,%s" % (_tex_escape(abbrev.rstrip(".")), _tex_escape(full))
            )
        else:
            parts.append(_tex_escape(full))
    return "; ".join(parts)


def _get_kn(by_relation: Dict[str, Any], model: str, rk: str) -> tuple[int, int]:
    bm = by_relation[rk]["by_model"]
    if model not in bm:
        return 0, 0
    stats = bm[model]
    return int(stats["n_relation_followed"]), int(stats["n_trials"])


def build_table(data: Dict[str, Any]) -> str:
    meta = data["meta"]
    by_relation: Dict[str, Any] = data["by_relation"]
    rel_keys = sorted(by_relation.keys(), key=lambda k: by_relation[k]["relation_name"])
    if not rel_keys:
        raise ValueError("No relations in JSON")

    models = _sort_models(list(by_relation[rel_keys[0]]["by_model"].keys()))
    n_rel = len(rel_keys)
    col_spec = (
        "@{}l@{\\hspace{4pt}}"
        + "@{\\hspace{2pt}}c" * n_rel
        + "|@{\\hspace{3pt}}c@{}"
    )

    header_cells = [
        "\\textbf{\\scriptsize " + _col_header_tex(rk, str(by_relation[rk]["relation_name"])) + "}"
        for rk in rel_keys
    ]
    header_cells.append("\\textbf{\\scriptsize Mean}")

    # Pooled column totals (over models) per relation
    col_kn: List[tuple[int, int]] = []
    for rk in rel_keys:
        sk, sn = 0, 0
        for model in models:
            k, n_t = _get_kn(by_relation, model, rk)
            sk += k
            sn += n_t
        col_kn.append((sk, sn))

    grand_k = sum(t[0] for t in col_kn)
    grand_n = sum(t[1] for t in col_kn)
    grand_p = grand_k / grand_n if grand_n else 0.0
    grand_se = _binomial_se(grand_k, grand_n)

    body_lines: List[str] = []
    for model in models:
        cells = [f"\\scriptsize {_tex_escape(_pretty_model(model))}"]
        row_k, row_n = 0, 0
        for rk in rel_keys:
            k, n_t = _get_kn(by_relation, model, rk)
            row_k += k
            row_n += n_t
            if n_t <= 0:
                cells.append("\\scriptsize ---")
                continue
            p = k / n_t
            se = _binomial_se(k, n_t)
            cells.append(f"\\scriptsize {_fmt_prop_se(p, se)}")
        rp = row_k / row_n if row_n else 0.0
        rse = _binomial_se(row_k, row_n)
        cells.append(
            "\\textbf{\\scriptsize " + _fmt_prop_se(rp, rse) + "}"
        )
        body_lines.append(" & ".join(cells) + " \\\\")

    bottom_cells = ["\\textbf{\\scriptsize Mean}"]
    for sk, sn in col_kn:
        cp = sk / sn if sn else 0.0
        cse = _binomial_se(sk, sn)
        bottom_cells.append(
            "\\textbf{\\scriptsize " + _fmt_prop_se(cp, cse) + "}"
        )
    bottom_cells.append(
        "\\textbf{\\scriptsize " + _fmt_prop_se(grand_p, grand_se) + "}"
    )
    bottom_row = " & ".join(bottom_cells) + " \\\\"

    ptf = meta.get("prompt_type_filter")
    ptf_note = (
        f"Prompt type restricted to \\texttt{{{_tex_escape(str(ptf))}}}."
        if ptf
        else "All prompt types included."
    )
    legend = _caption_column_legend(rel_keys, by_relation)
    caption = (
        "\\caption{LLM relation-following rates (binomial proportion with SE in parentheses) "
        "by model and relational category, from automated grading "
        "(\\texttt{judge\\_labels} / majority correct) in \\texttt{gold\\_curate\\_b} trials. "
        + ptf_note
        + " "
        "\\emph{Mean} column and row pool trials across relations or models, respectively; "
        "the bottom-right cell pools all trials. "
        "\\emph{Column labels:} "
        + legend
        + ".}"
    )

    lines: List[str] = [
        "% Requires: \\usepackage{booktabs}, \\usepackage{graphicx} (for \\resizebox).",
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{%s}" % col_spec,
        "\\toprule",
        "\\textbf{Model} & " + " & ".join(header_cells) + " \\\\",
        "\\midrule",
        *body_lines,
        "\\midrule",
        bottom_row,
        "\\bottomrule",
        "\\end{tabular}",
        "}%",
        caption,
        "\\label{tab:relation_following_validation}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not args.in_json.is_file():
        raise SystemExit(f"Input JSON not found: {args.in_json}")
    data = json.loads(args.in_json.read_text(encoding="utf-8"))
    tex = build_table(data)
    args.out_tex.write_text(tex, encoding="utf-8")
    print(f"Wrote {args.out_tex}")


if __name__ == "__main__":
    main()
