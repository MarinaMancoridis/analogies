from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate appendix LaTeX tables.")
    parser.add_argument(
        "--relations-csv",
        type=Path,
        default=here
        / "LLM_relations_completions"
        / "qualtrics_loop_and_merge"
        / "relation_loop_and_merge_all.csv",
    )
    parser.add_argument(
        "--identity-trials",
        type=Path,
        default=here / "LLM_identity_completions" / "runs" / "gold_run" / "trials.ndjson",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=here / "appendix_tables",
    )
    return parser.parse_args()


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def _relation_display_name(key: str) -> str:
    return key.replace("_", " ").title()


def build_relations_sample_table(relations_csv: Path) -> str:
    by_relation: Dict[str, Dict[str, str]] = {}
    with relations_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel = (row.get("relation_key") or "").strip()
            if rel and rel not in by_relation:
                by_relation[rel] = row

    rows: List[List[str]] = []
    for rel in sorted(by_relation.keys()):
        r = by_relation[rel]
        rows.append(
            [
                _latex_escape(_relation_display_name(rel)),
                _latex_escape((r.get("A") or "").strip()),
                _latex_escape((r.get("B") or "").strip()),
                _latex_escape((r.get("C") or "").strip()),
            ]
        )

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\small",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\begin{tabular}{llll}",
        "\\toprule",
        "Relation & A & B & C \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{One example triple per relation type. A and C are sampled from each relation's curated word list (\\texttt{curated\\_words.json}); B is author-generated.}",
            "\\label{tab:appendix_relation_examples}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def build_identity_metadata_table(identity_trials: Path, n_rows: int = 10) -> str:
    by_triple: Dict[int, Dict[str, Any]] = {}
    with identity_trials.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if str(row.get("prompt_type", "")) != "colon":
                continue
            tid = int(row.get("triple_id", -1))
            if tid < 0 or tid in by_triple:
                continue
            by_triple[tid] = row

    rows: List[List[str]] = []
    for tid in sorted(by_triple.keys())[:n_rows]:
        r = by_triple[tid]
        am = r.get("A_metrics", {}) or {}
        cm = r.get("C_metrics", {}) or {}
        rows.append(
            [
                str(tid),
                _latex_escape(str(r.get("A", ""))),
                _latex_escape(str(r.get("C", ""))),
                _latex_escape(str(am.get("pos", ""))),
                _latex_escape(str(cm.get("pos", ""))),
                f"{float(am.get('pop_zipf', 0.0)):.2f}",
                f"{float(cm.get('pop_zipf', 0.0)):.2f}",
                str(int(am.get("polysemy", 0) or 0)),
                str(int(cm.get("polysemy", 0) or 0)),
            ]
        )

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\footnotesize",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\begin{tabular}{rllccrrrr}",
        "\\toprule",
        "ID & First & Second & POS$_1$ & POS$_2$ & Freq$_1$ & Freq$_2$ & Sense$_1$ & Sense$_2$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Ten sample identity-task stimuli with lexical metadata. POS is universal part-of-speech tags (NLTK). Frequency is the English Zipf frequency from \\texttt{wordfreq}: $\\mathrm{freq}(w)=\\log_{10}(\\text{occurrences per billion tokens of }w)$. Senses is the WordNet synset count for the word.}",
            "\\label{tab:appendix_identity_samples_metadata}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rel_tex = build_relations_sample_table(args.relations_csv)
    id_tex = build_identity_metadata_table(args.identity_trials, n_rows=10)

    rel_out = args.out_dir / "appendix_relation_examples.tex"
    id_out = args.out_dir / "appendix_identity_samples_metadata.tex"
    rel_out.write_text(rel_tex + "\n", encoding="utf-8")
    id_out.write_text(id_tex + "\n", encoding="utf-8")

    print(f"Wrote: {rel_out}")
    print(f"Wrote: {id_out}")


if __name__ == "__main__":
    main()
