from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import random
import re
import shlex
import subprocess
import sys
import time
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
    "openai/gpt-5.4-mini",
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


def _slugify_model(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _append_ndjson_locked(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lock_path = path + ".lock"
    with open(lock_path, "w", encoding="utf-8") as lockf:
        import fcntl

        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        finally:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)


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


def _count_ndjson_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _print_progress(done: int, total: int) -> None:
    width = 30
    if total <= 0:
        total = 1
    frac = min(1.0, done / total)
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\rProgress [{bar}] {done}/{total} ({frac * 100:5.1f}%)", end="", flush=True)


def _print_worker_progress(model: str, done: int, total: int, hits: int) -> None:
    width = 30
    if total <= 0:
        total = 1
    frac = min(1.0, done / total)
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    acc = (hits / done) if done else 0.0
    label = _slugify_model(model)[:28]
    print(
        f"\r[{label}] [{bar}] {done}/{total} ({frac * 100:5.1f}%)  acc={acc:.3f} ({hits}/{done})",
        end="",
        flush=True,
    )


def _run_worker_for_model(
    *,
    run_dir: str,
    model: str,
    selected_triples: List[Dict[str, Any]],
    n_iters: int,
    run_mode: str,
) -> Dict[str, Any]:
    trials_path = os.path.join(run_dir, "trials.ndjson")
    counts = Counts()
    total_jobs = n_iters * len(selected_triples)
    _print_worker_progress(model, 0, total_jobs, 0)

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


def _worker_done_path(run_dir: str, model: str) -> str:
    return os.path.join(run_dir, "worker_done", f"{_slugify_model(model)}.json")


def _launch_model_in_new_terminal(
    *,
    model: str,
    run_dir: str,
    python_exe: str,
) -> None:
    script_path = os.path.abspath(__file__)
    cmd = " ".join(
        [
            shlex.quote(python_exe),
            shlex.quote(script_path),
            "--worker",
            "--model",
            shlex.quote(model),
            "--run-dir",
            shlex.quote(run_dir),
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


def _aggregate_summary(run_dir: str) -> Dict[str, Any]:
    manifest = _read_json(os.path.join(run_dir, "run_manifest.json"))
    per_model: Dict[str, Any] = {}
    overall = Counts()
    for model in manifest["models"]:
        done_path = _worker_done_path(run_dir, model)
        if not os.path.exists(done_path):
            continue
        stats = _read_json(done_path)
        per_model[model] = stats
        overall.n += int(stats["n"])
        overall.hits += int(stats["hits"])
        overall.errors += int(stats["errors"])

    summary = {
        **manifest,
        "per_model": per_model,
        "overall": {
            "n": overall.n,
            "hits": overall.hits,
            "errors": overall.errors,
            "success_rate": overall.hits / overall.n if overall.n else 0.0,
            "success_fraction": f"{overall.hits}/{overall.n}",
        },
    }
    _write_json(os.path.join(run_dir, "run_summary.json"), summary)
    return summary


def run_experiment(*, run_mode: str, seed: int, shard: bool) -> str:
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
    os.makedirs(os.path.join(out_dir, "worker_done"), exist_ok=True)
    trials_path = os.path.join(out_dir, "trials.ndjson")

    manifest: Dict[str, Any] = {
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
        "selected_triples": selected_triples,
        "sharded": shard,
    }
    _write_json(os.path.join(out_dir, "run_manifest.json"), manifest)
    open(trials_path, "a", encoding="utf-8").close()

    if not shard:
        for model in MODELS:
            stats = _run_worker_for_model(
                run_dir=out_dir,
                model=model,
                selected_triples=selected_triples,
                n_iters=n_iters,
                run_mode=run_mode,
            )
            _write_json(_worker_done_path(out_dir, model), stats)
            print(f"[{model}] success={stats['success_fraction']} errors={stats['errors']}")
        summary = _aggregate_summary(out_dir)
        print(f"\nRun dir: {out_dir}")
        print(f"Total generations: {summary['overall']['n']}")
        return out_dir

    python_exe = sys.executable
    for model in MODELS:
        _launch_model_in_new_terminal(model=model, run_dir=out_dir, python_exe=python_exe)

    total = manifest["expected_generations"]
    print(f"Run dir: {out_dir}")
    print("Spawned one terminal window per model. Waiting for completion...")
    while True:
        done = _count_ndjson_lines(trials_path)
        _print_progress(done, total)
        done_workers = sum(1 for m in MODELS if os.path.exists(_worker_done_path(out_dir, m)))
        if done_workers == len(MODELS):
            break
        time.sleep(2)
    print()

    summary = _aggregate_summary(out_dir)
    print(f"Completed. Total generations: {summary['overall']['n']}")
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
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Run models sequentially in a single process (default is sharded windows).",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Internal mode: run a single model worker.",
    )
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--run-dir", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.worker:
        if not args.model or not args.run_dir:
            raise ValueError("--worker requires --model and --run-dir")
        manifest = _read_json(os.path.join(args.run_dir, "run_manifest.json"))
        stats = _run_worker_for_model(
            run_dir=args.run_dir,
            model=args.model,
            selected_triples=manifest["selected_triples"],
            n_iters=int(manifest["iterations_per_model_per_triple"]),
            run_mode=str(manifest["run_mode"]),
        )
        _write_json(_worker_done_path(args.run_dir, args.model), stats)
        print(f"Worker done: {args.model} -> {stats['success_fraction']} errors={stats['errors']}")
        return

    run_experiment(run_mode=args.mode, seed=args.seed, shard=not args.serial)


if __name__ == "__main__":
    main()
