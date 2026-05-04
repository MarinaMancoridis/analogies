"""
Collect identity analogies completions for english / one_shot / few_shot prompts.

Uses the **same** `selected_triples` as an existing colon run (default: `runs/gold_run`),
one iteration per (model, triple, prompt_type)—no multi-iteration repeats.

Prompt text matches `analogy_types/structural/identity.py` PROMPT_VARIATIONS (same as the
structural identity task). Trials are appended in the same NDJSON shape as
`run_identity_triples.py` (iteration=1, `is_identity` = parsed answer == expected C).

Examples:
  # All models, serial (single process)
  python run_identity_prompt_variants.py --serial

  # One Terminal per model (default on macOS)
  python run_identity_prompt_variants.py

  # Worker (internal)
  python run_identity_prompt_variants.py --worker --model gpt-4o --run-dir runs/<id>
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Repo root is the `analogies` package (utils.py, constants.py at top level).
if "analogies" not in sys.modules:
    _pkg = types.ModuleType("analogies")
    _pkg.__path__ = [str(REPO_ROOT)]  # type: ignore[attr-defined]
    sys.modules["analogies"] = _pkg

from analogies.utils import generate_inference

# Reuse experiment id and model list from the main identity runner
from run_identity_triples import (
    EXPERIMENT_NAME,
    MODELS,
    _append_ndjson_locked,
    _aggregate_summary,
    _count_ndjson_lines,
    _launch_model_in_new_terminal,
    _make_run_dir,
    _print_progress,
    _print_worker_progress,
    _read_json,
    _slugify_model,
    _utc_now,
    _worker_done_path,
    _write_json,
)

RUNS_DIR = HERE / "runs"
DEFAULT_GOLD_RUN = HERE / "runs" / "gold_run"

NONCOLON_PROMPT_TYPES: List[str] = ["english", "one_shot", "few_shot"]

# Must stay aligned with analogy_types/structural/identity.py PROMPT_VARIATIONS.
_NONCOLON_PROMPT_BUILDERS = {
    "english": lambda A, C: (
        "Complete the analogy. Reply ONLY as: ANSWER: <word>\n\n"
        f"{A} is to {A} as {C} is to ____"
    ),
    "one_shot": lambda A, C: (
        "Complete the analogy. Reply ONLY as: ANSWER: <word>\n\n"
        f"{A} : {A} :: {C} : ____. \n\n"
        "Here is one example. \n"
        "rose : rose :: flower : flower"
    ),
    "few_shot": lambda A, C: (
        "Complete the analogy. Reply ONLY as: ANSWER: <word>\n\n"
        f"{A} : {A} :: {C} : ____. \n\n"
        "Here are some examples. \n"
        "rose : rose :: flower : flower \n"
        "apple : apple :: fruit : fruit \n"
        "cat : cat :: dog : dog"
    ),
}


@dataclass
class Counts:
    n: int = 0
    hits: int = 0
    errors: int = 0


def _clean_answer(resp: str) -> str:
    text = (resp or "").strip()
    m = re.search(r"ANSWER:\s*([A-Za-z][A-Za-z'-]*)", text, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    m2 = re.search(r"\b([A-Za-z][A-Za-z'-]*)\b", text)
    return m2.group(1).lower() if m2 else ""


def _prompt_for_type(prompt_type: str, a: str, c: str) -> str:
    fn = _NONCOLON_PROMPT_BUILDERS.get(prompt_type)
    if not callable(fn):
        raise KeyError(f"Unknown prompt_type {prompt_type!r}")
    return fn(a, c)


def _run_worker_for_model(
    *,
    run_dir: str,
    model: str,
    selected_triples: List[Dict[str, Any]],
    run_mode: str,
) -> Dict[str, Any]:
    trials_path = os.path.join(run_dir, "trials.ndjson")
    counts = Counts()
    total_jobs = len(selected_triples) * len(NONCOLON_PROMPT_TYPES)
    _print_worker_progress(model, 0, total_jobs, 0)

    for triple in selected_triples:
        a = str(triple["A"])
        b = str(triple["B"])
        c = str(triple["C"])
        expected = c
        for prompt_type in NONCOLON_PROMPT_TYPES:
            raw = ""
            parsed = ""
            error = None
            prompt = ""
            try:
                prompt = _prompt_for_type(prompt_type, a, c)
                raw = generate_inference(prompt, model)
                parsed = _clean_answer(raw)
                hit = parsed == expected.lower()
            except Exception as e:
                hit = False
                error = str(e)
                counts.errors += 1

            trial = {
                "timestamp_utc": _utc_now(),
                "experiment_name": EXPERIMENT_NAME,
                "run_mode": run_mode,
                "model": model,
                "iteration": 1,
                "triple_id": triple.get("triple_id"),
                "analogy_type": "identity",
                "prompt_type": prompt_type,
                "A": a,
                "B": b,
                "C": c,
                "expected": expected,
                "A_metrics": triple.get("A_metrics"),
                "B_metrics": triple.get("B_metrics"),
                "C_metrics": triple.get("C_metrics"),
                "prompt": prompt,
                "raw_response": raw,
                "parsed_answer": parsed,
                "is_identity": hit,
                "error": error,
            }
            _append_ndjson_locked(trials_path, trial)

            counts.n += 1
            counts.hits += int(hit)
            _print_worker_progress(model, counts.n, total_jobs, counts.hits)

    rate = counts.hits / counts.n if counts.n else 0.0
    print()
    return {
        "model": model,
        "n": counts.n,
        "hits": counts.hits,
        "errors": counts.errors,
        "success_rate": rate,
        "success_fraction": f"{counts.hits}/{counts.n}",
    }


def run_collection(
    *,
    gold_run_dir: Path,
    out_run_dir: Optional[Path],
    shard: bool,
    smoke: bool,
) -> str:
    manifest_path = gold_run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing run_manifest.json: {manifest_path}")

    gold_manifest = _read_json(str(manifest_path))
    selected: List[Dict[str, Any]] = gold_manifest.get("selected_triples") or []
    if not selected:
        raise ValueError(f"No selected_triples in {manifest_path}")

    if smoke:
        selected = selected[:1]

    models = MODELS[:1] if smoke else MODELS

    run_mode = "smoke_variants" if smoke else "noncolon_prompt_variants"
    if out_run_dir is not None:
        out_dir = str(out_run_dir.resolve())
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = _make_run_dir()

    os.makedirs(os.path.join(out_dir, "worker_done"), exist_ok=True)
    trials_path = os.path.join(out_dir, "trials.ndjson")

    expected_gens = len(models) * len(selected) * len(NONCOLON_PROMPT_TYPES)
    manifest: Dict[str, Any] = {
        "experiment_name": EXPERIMENT_NAME,
        "created_at_utc": _utc_now(),
        "run_mode": run_mode,
        "collection_kind": "identity_noncolon_prompts",
        "colon_reference_run_dir": str(gold_run_dir.resolve()),
        "colon_reference_manifest": str(manifest_path.resolve()),
        "prompt_types": NONCOLON_PROMPT_TYPES,
        "prompt_variations_source": "analogy_types/structural/identity.py PROMPT_VARIATIONS (non-colon)",
        "models": models,
        "n_models": len(models),
        "n_triples": len(selected),
        "iterations_per_model_per_triple": 1,
        "expected_generations": expected_gens,
        "selected_triples": selected,
        "sharded": shard,
    }
    _write_json(os.path.join(out_dir, "run_manifest.json"), manifest)
    open(trials_path, "a", encoding="utf-8").close()

    if not shard:
        for model in models:
            stats = _run_worker_for_model(
                run_dir=out_dir,
                model=model,
                selected_triples=selected,
                run_mode=run_mode,
            )
            _write_json(_worker_done_path(out_dir, model), stats)
            print(f"[{model}] success={stats['success_fraction']} errors={stats['errors']}")
        summary = _aggregate_summary(out_dir)
        print(f"\nRun dir: {out_dir}")
        print(f"Total generations: {summary['overall']['n']}")
        return out_dir

    python_exe = sys.executable
    script_path = os.path.abspath(__file__)
    for model in models:
        cmd = " ".join(
            [
                shlex.quote(python_exe),
                shlex.quote(script_path),
                "--worker",
                "--model",
                shlex.quote(model),
                "--run-dir",
                shlex.quote(out_dir),
            ]
        )
        if platform.system() == "Darwin":
            applescript = (
                'tell application "Terminal"\n'
                "activate\n"
                'set w to (do script "")\n'
                f'do script "cd {shlex.quote(os.getcwd())} && {cmd}" in w\n'
                "end tell"
            )
            subprocess.run(["osascript", "-e", applescript], check=True)
        else:
            subprocess.Popen(cmd, shell=True)

    print(f"Run dir: {out_dir}")
    print("Spawned one terminal window per model. Waiting for completion...")
    while True:
        done = _count_ndjson_lines(trials_path)
        _print_progress(done, expected_gens)
        done_workers = sum(1 for m in models if os.path.exists(_worker_done_path(out_dir, m)))
        if done_workers == len(models):
            break
        time.sleep(2)
    print()

    summary = _aggregate_summary(out_dir)
    print(f"Completed. Total generations: {summary['overall']['n']}")
    return out_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--gold-run-dir",
        type=Path,
        default=DEFAULT_GOLD_RUN,
        help="Directory with colon gold run (must contain run_manifest.json).",
    )
    p.add_argument(
        "--out-run-dir",
        type=Path,
        default=None,
        help="Write into this directory instead of a new timestamped runs/ folder.",
    )
    p.add_argument(
        "--serial",
        action="store_true",
        help="Run all models in this process (no per-model Terminal windows).",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="1 triple, 1 model, for a quick API check.",
    )
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--model", type=str, default="")
    p.add_argument("--run-dir", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.worker:
        if not args.model or not args.run_dir:
            raise SystemExit("--worker requires --model and --run-dir")
        manifest = _read_json(os.path.join(args.run_dir, "run_manifest.json"))
        stats = _run_worker_for_model(
            run_dir=args.run_dir,
            model=args.model,
            selected_triples=manifest["selected_triples"],
            run_mode=str(manifest["run_mode"]),
        )
        _write_json(_worker_done_path(args.run_dir, args.model), stats)
        print(f"Worker done: {args.model} -> {stats['success_fraction']} errors={stats['errors']}")
        return

    run_collection(
        gold_run_dir=args.gold_run_dir,
        out_run_dir=args.out_run_dir,
        shard=not args.serial,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
