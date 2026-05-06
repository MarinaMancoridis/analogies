#!/usr/bin/env python3
"""
Emit many (default 300) analogy blocks: stem, then all human Ds, then several model Ds.

Each block is three lines:
  1)  (RelationName) A : B :: C : ____
  2)  (indent) humans: every human completion D for this triple_id (same filter as --include-ineligible-humans).
  3)  (indent) models: up to N distinct models, each labeled, one D sampled if a model has multiple rows.

Run from repo root:
  python experiments/LLM_relations_completions/sample_prompt_llm_human.py
  python experiments/LLM_relations_completions/sample_prompt_llm_human.py --n 300 --seed 0 --n-models 3
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

INDENT = "        "


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_llm_trials(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("error"):
                continue
            rows.append(rec)
    return rows


def _load_humans(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("completions") or [])


def _index_llm_by_triple(
    trials: List[Dict[str, Any]], prompt_type: str
) -> Dict[int, List[Dict[str, Any]]]:
    by_tid: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for rec in trials:
        if rec.get("prompt_type") != prompt_type:
            continue
        tid = rec.get("triple_id")
        if tid is None:
            continue
        by_tid[int(tid)].append(rec)
    return dict(by_tid)


def _index_humans_by_triple(
    completions: List[Dict[str, Any]], eligible_only: bool
) -> Dict[int, List[Dict[str, Any]]]:
    by_tid: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for rec in completions:
        if eligible_only and not rec.get("participant_eligible_complete"):
            continue
        tid = rec.get("triple_id")
        if tid is None:
            continue
        by_tid[int(tid)].append(rec)
    return dict(by_tid)


def _relation_label(rec: Dict[str, Any]) -> str:
    name = str(rec.get("relation_name") or "").strip()
    if name:
        return name
    key = str(rec.get("relation_key") or "").strip()
    if key:
        return key.replace("_", " ").title()
    return "unknown"


def _analogy_stem(rec: Dict[str, Any]) -> str:
    a = str(rec.get("A") or "").strip()
    b = str(rec.get("B") or "").strip()
    c = str(rec.get("C") or "").strip()
    rel = _relation_label(rec)
    return f"({rel}) {a} : {b} :: {c} : ____"


def _format_humans_line(hum_rows: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for r in hum_rows:
        d = str(r.get("D") or "").strip()
        parts.append(d if d else "(empty)")
    if not parts:
        return f"{INDENT}humans: (none)"
    return f"{INDENT}humans: " + "; ".join(parts)


def _format_models_line(
    llm_rows: List[Dict[str, Any]],
    n_models: int,
    rng: random.Random,
) -> str:
    by_m: Dict[str, List[str]] = defaultdict(list)
    for r in llm_rows:
        m = str(r.get("model") or "").strip()
        if not m:
            continue
        d = str(r.get("D") or "").strip()
        by_m[m].append(d if d else "(empty)")

    if not by_m:
        return f"{INDENT}models: (none)"

    names = list(by_m.keys())
    k = min(max(1, n_models), len(names))
    if len(names) > k:
        chosen_models = rng.sample(names, k)
    else:
        chosen_models = names[:]
        rng.shuffle(chosen_models)

    chunks: List[str] = []
    for m in chosen_models:
        choices = by_m[m]
        pick = rng.choice(choices) if choices else "(empty)"
        chunks.append(f"{m}: {pick}")

    return f"{INDENT}models: " + "; ".join(chunks)


def parse_args() -> argparse.Namespace:
    root = _repo_root()
    here = Path(__file__).resolve().parent
    vf = root / "experiments" / "validate_relation_following" / "human_completions_flat.json"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--trials",
        type=Path,
        default=here / "runs" / "gold_curate_b" / "trials.ndjson",
        help="LLM relations trials.ndjson",
    )
    p.add_argument(
        "--humans-json",
        type=Path,
        default=vf,
        help="human_completions_flat.json from validate_human_responses.py",
    )
    p.add_argument(
        "--prompt-type",
        type=str,
        default="colon",
        choices=["colon", "english", "one_shot", "few_shot"],
        help="Which LLM trials to use when grouping by model (same triple_id, same prompt_type).",
    )
    p.add_argument(
        "--include-ineligible-humans",
        action="store_true",
        help="Include human rows even when participant_eligible_complete is false.",
    )
    p.add_argument("--n", type=int, default=300, help="How many analogy blocks to print.")
    p.add_argument(
        "--n-models",
        type=int,
        default=3,
        help="How many distinct LLM models to show per block (default 3).",
    )
    p.add_argument("--seed", type=int, default=None, help="RNG seed (optional).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    eligible_only = not bool(args.include_ineligible_humans)
    n_want = max(0, args.n)
    n_models = max(1, args.n_models)

    if not args.trials.is_file():
        raise SystemExit(f"Missing trials file: {args.trials}")
    if not args.humans_json.is_file():
        raise SystemExit(
            f"Missing human completions JSON: {args.humans_json}\n"
            "Generate it with:\n"
            "  python experiments/validate_relation_following/validate_human_responses.py"
        )

    rng = random.Random(args.seed)

    trials = _load_llm_trials(args.trials)
    humans = _load_humans(args.humans_json)

    llm_by_tid = _index_llm_by_triple(trials, args.prompt_type)
    hum_by_tid = _index_humans_by_triple(humans, eligible_only)

    common = list(set(llm_by_tid.keys()) & set(hum_by_tid.keys()))
    if not common:
        raise SystemExit(
            "No triple_id overlap between LLM trials and human completions "
            "(check paths, prompt_type, and eligibility filter)."
        )

    rng.shuffle(common)

    chosen_tids: List[int] = []
    if n_want <= len(common):
        chosen_tids = common[:n_want]
    else:
        chosen_tids = common[:]
        while len(chosen_tids) < n_want:
            chosen_tids.append(rng.choice(common))

    out_lines: List[str] = []
    for tid in chosen_tids:
        llm_rows = llm_by_tid[tid]
        hum_rows = hum_by_tid[tid]
        stem = _analogy_stem(llm_rows[0])

        out_lines.append(stem)
        out_lines.append(_format_humans_line(hum_rows))
        out_lines.append(_format_models_line(llm_rows, n_models, rng))

    print("\n".join(out_lines))


if __name__ == "__main__":
    main()
