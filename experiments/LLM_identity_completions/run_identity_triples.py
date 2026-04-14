from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List

from analogies.utils import generate_inference

EXPERIMENT_NAME = "LLM_identity_completions"
HERE = os.path.dirname(__file__)
RUNS_DIR = os.path.join(HERE, "runs")
TRIPLES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(HERE)),
    "static_triples",
    "identity",
    "identity_triples.json",
)

# 10-model set (configured in constants.py + generate_inference()).
MODELS: List[str] = [
    "meta-llama/llama-3.3-70b-instruct",
    "gpt-4o",
    "openai/gpt-4.1",
    "openai/gpt-5.4-pro",
    "google/gemini-3-flash-preview",
    "google/gemini-3.1-flash-lite-preview",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-opus-4.6",
    "deepseek-ai/DeepSeek-V3",
    "deepseek/deepseek-v3.2",
]

PROMPT_TYPE = "colon"
DEFAULT_SMOKE_TRIPLES = 1
DEFAULT_SMOKE_ITERS = 1
DEFAULT_FULL_TRIPLES = 100
DEFAULT_FULL_ITERS = 3


@dataclass
class Counts:
    n: int = 0
    hits: int = 0
    errors: int = 0


def _utc_now() -> str:
    return dt.datetime.utcnow().isoformat() + "Z"


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _append_ndjson(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _load_triples(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    triples = payload.get("triples", [])
    if not triples:
        raise ValueError(f"No triples found in {path}")
    return triples


def _build_prompt(a: str, b: str, c: str) -> str:
    return (
        "Complete the analogy. Reply ONLY as: ANSWER: <word>\n\n"
        f"{a} : {b} :: {c} : ____"
    )


def _clean_answer(resp: str) -> str:
    text = (resp or "").strip()
    m = re.search(r"ANSWER:\s*([A-Za-z][A-Za-z'-]*)", text, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    m2 = re.search(r"\b([A-Za-z][A-Za-z'-]*)\b", text)
    return m2.group(1).lower() if m2 else ""


def _make_run_dir() -> str:
    run_id = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    out_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _print_progress(done: int, total: int) -> None:
    width = 30
    if total <= 0:
        total = 1
    frac = done / total
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\rProgress [{bar}] {done}/{total} ({frac * 100:5.1f}%)", end="", flush=True)


def run_experiment(*, run_mode: str, seed: int) -> str:
    rng = random.Random(seed)
    triples_all = _load_triples(TRIPLES_PATH)

    if run_mode == "smoke":
        n_triples = min(DEFAULT_SMOKE_TRIPLES, len(triples_all))
        n_iters = DEFAULT_SMOKE_ITERS
    else:
        n_triples = min(DEFAULT_FULL_TRIPLES, len(triples_all))
        n_iters = DEFAULT_FULL_ITERS

    selected_triples = rng.sample(triples_all, n_triples)
    out_dir = _make_run_dir()
    trials_path = os.path.join(out_dir, "trials.ndjson")

    summary: Dict[str, Any] = {
        "experiment_name": EXPERIMENT_NAME,
        "created_at_utc": _utc_now(),
        "run_mode": run_mode,
        "seed": seed,
        "prompt_type": PROMPT_TYPE,
        "triples_source": TRIPLES_PATH,
        "models": MODELS,
        "n_models": len(MODELS),
        "n_triples": n_triples,
        "iterations_per_model_per_triple": n_iters,
        "expected_generations": len(MODELS) * n_triples * n_iters,
        "per_model": {},
    }

    overall = Counts()
    total_jobs = len(MODELS) * n_triples * n_iters
    _print_progress(0, total_jobs)

    for model in MODELS:
        counts = Counts()
        for iteration in range(1, n_iters + 1):
            for triple in selected_triples:
                a = triple["A"]
                b = triple["B"]
                c = triple["C"]
                expected = c
                prompt = _build_prompt(a, b, c)
                raw = ""
                parsed = ""
                error = None

                try:
                    raw = generate_inference(prompt, model)
                    parsed = _clean_answer(raw)
                    hit = parsed == expected.lower()
                except Exception as e:
                    hit = False
                    error = str(e)
                    counts.errors += 1
                    overall.errors += 1

                trial = {
                    "timestamp_utc": _utc_now(),
                    "experiment_name": EXPERIMENT_NAME,
                    "run_mode": run_mode,
                    "model": model,
                    "iteration": iteration,
                    "triple_id": triple.get("triple_id"),
                    "analogy_type": "identity",
                    "prompt_type": PROMPT_TYPE,
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
                _append_ndjson(trials_path, trial)

                counts.n += 1
                counts.hits += int(hit)
                overall.n += 1
                overall.hits += int(hit)
                _print_progress(overall.n, total_jobs)

        rate = counts.hits / counts.n if counts.n else 0.0
        print()
        summary["per_model"][model] = {
            "n": counts.n,
            "hits": counts.hits,
            "errors": counts.errors,
            "success_rate": rate,
            "success_fraction": f"{counts.hits}/{counts.n}",
        }
        print(f"[{model}] success={counts.hits}/{counts.n} ({rate:.3f}), errors={counts.errors}")

    summary["overall"] = {
        "n": overall.n,
        "hits": overall.hits,
        "errors": overall.errors,
        "success_rate": overall.hits / overall.n if overall.n else 0.0,
        "success_fraction": f"{overall.hits}/{overall.n}",
    }
    summary["selected_triples"] = selected_triples

    _write_json(os.path.join(out_dir, "run_summary.json"), summary)
    print(f"\nRun dir: {out_dir}")
    print(f"Total generations: {overall.n}")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run identity completion on static triples across 10 models."
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "full"],
        default="smoke",
        help="smoke = 1 triple x 10 models x 1 iter; full = 100 triples x 10 models x 3 iters",
    )
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(run_mode=args.mode, seed=args.seed)


if __name__ == "__main__":
    main()
