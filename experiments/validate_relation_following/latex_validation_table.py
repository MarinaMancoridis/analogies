from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    p.add_argument(
        "--human-panel",
        type=Path,
        default=here / "human_judgments_panel.ndjson",
        help=(
            "NDJSON of human completion judgments (three judge rows per completion_id). "
            "Used for the Human table row (≥2/3 judges Correct = relation-followed)."
        ),
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


def _human_panel_majority_by_relation(
    panel_path: Path,
) -> Tuple[Dict[str, Tuple[int, int]], Tuple[int, int]]:
    """
    One vote per completion_id: relation-followed iff ≥2 of 3 panel judges have
    judge_majority_correct True (each NDJSON row is one judge).
    """
    votes: Dict[str, List[bool]] = defaultdict(list)
    cid_rel: Dict[str, str] = {}
    with panel_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("record_type") != "human_completion_judgment":
                continue
            cid = rec.get("completion_id")
            if not isinstance(cid, str):
                continue
            votes[cid].append(rec.get("judge_majority_correct") is True)
            cid_rel[cid] = str(rec.get("relation_key", ""))

    by_rel: Dict[str, Tuple[int, int]] = {}
    grand_k, grand_n = 0, 0
    for rk in set(cid_rel.values()):
        if rk:
            by_rel[rk] = (0, 0)

    for cid, ys in votes.items():
        if len(ys) != 3:
            continue
        rk = cid_rel.get(cid, "")
        if not rk or rk not in by_rel:
            continue
        maj_correct = sum(ys) >= 2
        k, n = by_rel[rk]
        by_rel[rk] = (k + (1 if maj_correct else 0), n + 1)
        grand_k += 1 if maj_correct else 0
        grand_n += 1

    return by_rel, (grand_k, grand_n)


def build_table(
    data: Dict[str, Any],
    human_by_rel: Optional[Dict[str, Tuple[int, int]]] = None,
    human_grand: Optional[Tuple[int, int]] = None,
) -> str:
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

    footer_lines: List[str] = []
    if human_by_rel is not None and human_grand is not None:
        h_cells = ["\\textbf{\\scriptsize Human}"]
        hgk, hgn = human_grand
        for rk in rel_keys:
            k, n = human_by_rel.get(rk, (0, 0))
            if n <= 0:
                h_cells.append("\\scriptsize ---")
                continue
            hp = k / n
            hse = _binomial_se(k, n)
            h_cells.append("\\textbf{\\scriptsize " + _fmt_prop_se(hp, hse) + "}")
        hgp = hgk / hgn if hgn else 0.0
        hgse = _binomial_se(hgk, hgn)
        h_cells.append("\\textbf{\\scriptsize " + _fmt_prop_se(hgp, hgse) + "}")
        footer_lines.append(" & ".join(h_cells) + " \\\\")

    ptf = meta.get("prompt_type_filter")
    ptf_note = (
        f"Prompt type restricted to \\texttt{{{_tex_escape(str(ptf))}}}."
        if ptf
        else "All prompt types included."
    )
    legend = _caption_column_legend(rel_keys, by_relation)
    human_note = ""
    if human_by_rel is not None:
        human_note = (
            "\\emph{Human} row: eligible human completions; each cell is the fraction for which "
            "\\(\\geq 2/3\\) LLM judges (same panel as validation) assigned \\texttt{Correct}. "
        )
    caption = (
        "\\caption{LLM relation-following rates (binomial proportion with SE in parentheses) "
        "by model and relational category, from automated grading "
        "(\\texttt{judge\\_labels} / majority correct) in \\texttt{gold\\_curate\\_b} trials. "
        + ptf_note
        + " "
        + "\\emph{Mean} column pools trials across relations for each model. "
        + human_note
        + "\\emph{Column labels:} "
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
        *footer_lines,
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
    human_by_rel: Optional[Dict[str, Tuple[int, int]]] = None
    human_grand: Optional[Tuple[int, int]] = None
    if args.human_panel.is_file():
        human_by_rel, human_grand = _human_panel_majority_by_relation(args.human_panel)
        print(
            f"Human panel: {human_grand[0]}/{human_grand[1]} majority-correct completions "
            f"({args.human_panel})",
            flush=True,
        )
    else:
        raise SystemExit(
            f"Human panel NDJSON not found (required for Human row): {args.human_panel}"
        )
    tex = build_table(data, human_by_rel=human_by_rel, human_grand=human_grand)
    args.out_tex.write_text(tex, encoding="utf-8")
    print(f"Wrote {args.out_tex}")


if __name__ == "__main__":
    main()
