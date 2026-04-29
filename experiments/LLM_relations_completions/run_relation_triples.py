from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
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
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Tuple


EXPERIMENT_NAME = "LLM_relations_completions"
HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"
RELATIONAL_DIR = HERE.parent.parent / "analogy_types" / "relational"
TRIPLES_PATH = HERE.parent.parent / "static_triples" / "relational" / "relation_triples.json"
FIXED_B_CSV_PATH = HERE / "qualtrics_loop_and_merge" / "relation_loop_and_merge_all.csv"
REPO_ROOT = HERE.parent.parent
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

# Match the 10-model set used in the identity experiment.
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
SMOKE_MODELS: List[str] = ["gpt-4o"]

PROMPT_TYPES = ["colon", "english", "one_shot", "few_shot"]
DEFAULT_SMOKE_TRIPLES_PER_RELATION = 1
DEFAULT_FULL_TRIPLES_PER_RELATION = 50
DEFAULT_JUDGE_N = 1


@dataclass
class Counts:
    n: int = 0
    b_nonempty: int = 0
    d_nonempty: int = 0
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


def _count_ndjson_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _print_progress(done: int, total: int) -> None:
    width = 30
    total = max(total, 1)
    frac = min(1.0, done / total)
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\rProgress [{bar}] {done}/{total} ({frac * 100:5.1f}%)", end="", flush=True)


def _print_worker_progress(model: str, done: int, total: int, d_nonempty: int) -> None:
    width = 30
    total = max(total, 1)
    frac = min(1.0, done / total)
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    rate = d_nonempty / done if done else 0.0
    label = _slugify_model(model)[:28]
    print(
        f"\r[{label}] [{bar}] {done}/{total} ({frac * 100:5.1f}%)  nonempty_D={rate:.3f} ({d_nonempty}/{done})",
        end="",
        flush=True,
    )


def _make_run_dir() -> str:
    run_id = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    out_dir = RUNS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir)


def _discover_relation_dirs() -> List[Path]:
    relation_dirs: List[Path] = []
    for child in sorted(RELATIONAL_DIR.iterdir()):
        if not child.is_dir():
            continue
        if (child / "curated_words.json").exists() and (child / f"{child.name}.py").exists():
            relation_dirs.append(child)
    if len(relation_dirs) != 10:
        raise ValueError(
            f"Expected 10 relational folders with curated_words.json + module, found {len(relation_dirs)}"
        )
    return relation_dirs


_MODULE_CACHE: Dict[str, ModuleType] = {}


def _load_relation_module(relation_key: str) -> ModuleType:
    if relation_key in _MODULE_CACHE:
        return _MODULE_CACHE[relation_key]

    module_path = RELATIONAL_DIR / relation_key / f"{relation_key}.py"
    spec = importlib.util.spec_from_file_location(f"relation_{relation_key}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load relation module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MODULE_CACHE[relation_key] = module
    return module


def _load_curated_words(path: Path) -> List[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    words = payload.get("words", [])
    words = [w.strip().lower() for w in words if isinstance(w, str) and w.strip()]
    if len(words) < 2:
        raise ValueError(f"Need at least 2 words in {path}, found {len(words)}")
    return words


def _word_metrics_or_none(module: ModuleType, word: str) -> Any:
    """Return relation-specific word metrics when available."""
    if not word:
        return None
    fn = getattr(module, "word_metrics", None)
    if callable(fn):
        try:
            return fn(word)
        except Exception:
            return None
    return None


def _run_relation_judge(module: ModuleType, model: str, a: str, b: str, c: str, d: str) -> Dict[str, Any]:
    """Run relation-specific majority judge helper when available."""
    judge_fn = getattr(module, "_majority_judge", None)
    if not callable(judge_fn):
        return {
            "judge_prompt": "",
            "judge_raws": [],
            "judge_labels": [],
            "judge_counts": {},
            "judge_majority_label": None,
            "judge_majority_correct": False,
            "judge_n": 0,
            "judge_error": "module missing _majority_judge",
        }

    try:
        out = judge_fn(model=model, A=a, B=b, C=c, D=d, n_judges=DEFAULT_JUDGE_N)
    except TypeError:
        try:
            out = judge_fn(model=model, A=a, B=b, C=c, D=d, n=DEFAULT_JUDGE_N)
        except TypeError:
            out = judge_fn(model, a, b, c, d, DEFAULT_JUDGE_N)
    except Exception as e:
        return {
            "judge_prompt": "",
            "judge_raws": [],
            "judge_labels": [],
            "judge_counts": {},
            "judge_majority_label": None,
            "judge_majority_correct": False,
            "judge_n": 0,
            "judge_error": str(e),
        }

    if not isinstance(out, dict):
        return {
            "judge_prompt": "",
            "judge_raws": [],
            "judge_labels": [],
            "judge_counts": {},
            "judge_majority_label": None,
            "judge_majority_correct": False,
            "judge_n": 0,
            "judge_error": "judge output was not a dict",
        }

    return {
        "judge_prompt": out.get("judge_prompt", ""),
        "judge_raws": out.get("judge_raws", []) or [],
        "judge_labels": out.get("judge_labels", []) or [],
        "judge_counts": out.get("judge_counts", {}) or {},
        "judge_majority_label": out.get("judge_majority_label"),
        "judge_majority_correct": bool(out.get("judge_majority_correct", False)),
        "judge_n": int(out.get("judge_n", 0) or 0),
        "judge_error": "",
    }


def _load_precomputed_triples(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    triples = payload.get("triples", [])
    if not isinstance(triples, list) or not triples:
        raise ValueError(f"No triples found in {path}")
    return triples


def _build_selected_triples(*, n_per_relation: int) -> List[Dict[str, Any]]:
    precomputed = _load_precomputed_triples(TRIPLES_PATH)
    relation_dirs = {d.name: d for d in _discover_relation_dirs()}
    by_relation: Dict[str, List[Dict[str, Any]]] = {}
    for t in precomputed:
        rel = str(t["relation_key"])
        by_relation.setdefault(rel, []).append(t)

    selected: List[Dict[str, Any]] = []
    for relation_key in sorted(relation_dirs.keys()):
        triples = sorted(by_relation.get(relation_key, []), key=lambda x: int(x["relation_triple_id"]))
        if len(triples) < n_per_relation:
            raise ValueError(
                f"Need at least {n_per_relation} triples for relation '{relation_key}' in {TRIPLES_PATH}, "
                f"found {len(triples)}"
            )
        module = _load_relation_module(relation_key)
        relation_dir = relation_dirs[relation_key]
        for t in triples[:n_per_relation]:
            selected.append(
                {
                    **t,
                    "relation_name": getattr(module, "REL_NAME", relation_key),
                    "relation_text": getattr(module, "REL_TEXT", ""),
                    "module_path": str(relation_dir / f"{relation_key}.py"),
                }
            )
    return selected


def _load_fixed_b_map(path: Path) -> Dict[int, Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Fixed-B CSV not found: {path}")
    by_id: Dict[int, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"triple_id", "relation_key", "A", "B", "C"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Fixed-B CSV missing columns: {sorted(missing)}")
        for row in reader:
            tid_raw = (row.get("triple_id") or "").strip()
            if not tid_raw:
                continue
            tid = int(tid_raw)
            by_id[tid] = {
                "relation_key": (row.get("relation_key") or "").strip(),
                "A": (row.get("A") or "").strip(),
                "B": (row.get("B") or "").strip(),
                "C": (row.get("C") or "").strip(),
            }
    if not by_id:
        raise ValueError(f"No rows found in fixed-B CSV: {path}")
    return by_id


def _attach_fixed_b(
    *,
    selected_triples: List[Dict[str, Any]],
    fixed_b_map: Dict[int, Dict[str, str]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for t in selected_triples:
        tid = int(t["triple_id"])
        row = fixed_b_map.get(tid)
        if row is None:
            raise ValueError(f"Fixed-B CSV missing triple_id={tid}")
        relation_key = str(t["relation_key"])
        a = str(t["A"])
        c = str(t["C"])
        if row["relation_key"] != relation_key or row["A"] != a or row["C"] != c:
            raise ValueError(
                f"Fixed-B mismatch for triple_id={tid}: "
                f"expected ({relation_key}, {a}, {c}) got ({row['relation_key']}, {row['A']}, {row['C']})"
            )
        out.append({**t, "B_fixed": row["B"]})
    return out


def _run_worker_for_model(
    *,
    run_dir: str,
    model: str,
    selected_triples: List[Dict[str, Any]],
    run_mode: str,
) -> Dict[str, Any]:
    trials_path = os.path.join(run_dir, "trials.ndjson")
    counts = Counts()
    total_jobs = len(selected_triples) * len(PROMPT_TYPES)
    _print_worker_progress(model, 0, total_jobs, 0)

    for triple in selected_triples:
        relation_key = str(triple["relation_key"])
        module = _load_relation_module(relation_key)
        a = str(triple["A"])
        c = str(triple["C"])
        for prompt_type in PROMPT_TYPES:
            error = None
            b = ""
            d = ""
            raw_b = ""
            raw_d = ""
            prompt_b = ""
            prompt_d = ""
            judge: Dict[str, Any] = {
                "judge_prompt": "",
                "judge_raws": [],
                "judge_labels": [],
                "judge_counts": {},
                "judge_majority_label": None,
                "judge_majority_correct": False,
                "judge_n": 0,
                "judge_error": "",
            }

            try:
                if "B_fixed" in triple:
                    b = str(triple.get("B_fixed", ""))
                    prompt_b = "[fixed_b_from_csv]"
                    raw_b = f"ANSWER: {b}" if b else "ANSWER: "
                else:
                    prompt_b = module._prompt_generate_B(a)
                    b, raw_b = module._ask_one(model, prompt_b)
                prompt_d = module.PROMPT_VARIATIONS[prompt_type](a, b, c)
                d, raw_d = module._ask_one(model, prompt_d)
                judge = _run_relation_judge(module, model, a, b, c, d)
            except Exception as e:
                error = str(e)
                counts.errors += 1

            trial = {
                "timestamp_utc": _utc_now(),
                "experiment_name": EXPERIMENT_NAME,
                "run_mode": run_mode,
                "model": model,
                "relation_key": relation_key,
                "relation_name": triple["relation_name"],
                "relation_text": triple["relation_text"],
                "triple_id": triple["triple_id"],
                "relation_triple_id": triple["relation_triple_id"],
                "prompt_type": prompt_type,
                "A": a,
                "B": b,
                "C": c,
                "D": d,
                "A_metrics": _word_metrics_or_none(module, a),
                "B_metrics": _word_metrics_or_none(module, b),
                "C_metrics": _word_metrics_or_none(module, c),
                "D_metrics": _word_metrics_or_none(module, d),
                "prompt_generate_B": prompt_b,
                "prompt_complete": prompt_d,
                "raw_response_B": raw_b,
                "raw_response_D": raw_d,
                "grade_prompt": judge["judge_prompt"],
                "grade_raw": judge["judge_raws"][-1] if judge["judge_raws"] else "",
                "grade_correct": judge["judge_majority_correct"],
                "judge_raws": judge["judge_raws"],
                "judge_labels": judge["judge_labels"],
                "judge_counts": judge["judge_counts"],
                "judge_majority_label": judge["judge_majority_label"],
                "judge_majority_correct": judge["judge_majority_correct"],
                "judge_n": judge["judge_n"],
                "judge_error": judge["judge_error"],
                "error": error,
            }
            _append_ndjson_locked(trials_path, trial)

            counts.n += 1
            counts.b_nonempty += int(bool(b))
            counts.d_nonempty += int(bool(d))
            _print_worker_progress(model, counts.n, total_jobs, counts.d_nonempty)

    print()
    return {
        "model": model,
        "n": counts.n,
        "b_nonempty": counts.b_nonempty,
        "d_nonempty": counts.d_nonempty,
        "errors": counts.errors,
        "d_nonempty_rate": counts.d_nonempty / counts.n if counts.n else 0.0,
    }


def _worker_done_path(run_dir: str, model: str) -> str:
    return os.path.join(run_dir, "worker_done", f"{_slugify_model(model)}.json")


def _launch_model_in_new_terminal(*, model: str, run_dir: str, python_exe: str) -> None:
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
        overall.b_nonempty += int(stats["b_nonempty"])
        overall.d_nonempty += int(stats["d_nonempty"])
        overall.errors += int(stats["errors"])

    summary = {
        **manifest,
        "per_model": per_model,
        "overall": {
            "n": overall.n,
            "b_nonempty": overall.b_nonempty,
            "d_nonempty": overall.d_nonempty,
            "errors": overall.errors,
            "d_nonempty_rate": overall.d_nonempty / overall.n if overall.n else 0.0,
        },
    }
    _write_json(os.path.join(run_dir, "run_summary.json"), summary)
    return summary


def run_experiment(*, run_mode: str, seed: int, shard: bool, fixed_b_csv: str = "") -> str:
    if run_mode == "smoke":
        n_per_relation = DEFAULT_SMOKE_TRIPLES_PER_RELATION
        models = SMOKE_MODELS
    else:
        n_per_relation = DEFAULT_FULL_TRIPLES_PER_RELATION
        models = MODELS

    selected_triples = _build_selected_triples(n_per_relation=n_per_relation)
    fixed_b_source = ""
    if fixed_b_csv:
        fixed_b_source = str(Path(fixed_b_csv).resolve())
        selected_triples = _attach_fixed_b(
            selected_triples=selected_triples,
            fixed_b_map=_load_fixed_b_map(Path(fixed_b_source)),
        )
    out_dir = _make_run_dir()
    os.makedirs(os.path.join(out_dir, "worker_done"), exist_ok=True)
    trials_path = os.path.join(out_dir, "trials.ndjson")

    manifest: Dict[str, Any] = {
        "experiment_name": EXPERIMENT_NAME,
        "created_at_utc": _utc_now(),
        "run_mode": run_mode,
        "seed": seed,
        "prompt_types": PROMPT_TYPES,
        "relations_source_dir": str(RELATIONAL_DIR),
        "triples_source": str(TRIPLES_PATH),
        "fixed_b_source": fixed_b_source,
        "models": models,
        "n_models": len(models),
        "n_relations": len(_discover_relation_dirs()),
        "triples_per_relation": n_per_relation,
        "n_total_relation_triples": len(selected_triples),
        "expected_generations": len(models) * len(selected_triples) * len(PROMPT_TYPES),
        "selected_triples": selected_triples,
        "sharded": shard,
    }
    _write_json(os.path.join(out_dir, "run_manifest.json"), manifest)
    open(trials_path, "a", encoding="utf-8").close()

    if not shard:
        for model in models:
            stats = _run_worker_for_model(
                run_dir=out_dir,
                model=model,
                selected_triples=selected_triples,
                run_mode=run_mode,
            )
            _write_json(_worker_done_path(out_dir, model), stats)
            print(f"[{model}] D_nonempty={stats['d_nonempty']}/{stats['n']} errors={stats['errors']}")
        summary = _aggregate_summary(out_dir)
        print(f"\nRun dir: {out_dir}")
        print(f"Total generations: {summary['overall']['n']}")
        return out_dir

    python_exe = sys.executable
    for model in models:
        _launch_model_in_new_terminal(model=model, run_dir=out_dir, python_exe=python_exe)

    total = manifest["expected_generations"]
    print(f"Run dir: {out_dir}")
    print("Spawned one terminal window per model. Waiting for completion...")
    while True:
        done = _count_ndjson_lines(trials_path)
        _print_progress(done, total)
        done_workers = sum(1 for m in models if os.path.exists(_worker_done_path(out_dir, m)))
        if done_workers == len(models):
            break
        time.sleep(2)
    print()

    summary = _aggregate_summary(out_dir)
    print(f"Completed. Total generations: {summary['overall']['n']}")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run relational completion generation across the 10 analogy relations."
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "full"],
        default="smoke",
        help=(
            "smoke = 1 triple per relation x 1 model; "
            "full = 50 triples per relation x 10 models x 4 prompt types"
        ),
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--fixed-b-csv",
        type=str,
        default="",
        help=(
            "Optional CSV with columns triple_id,relation_key,A,B,C. "
            "When provided, B is deterministic from CSV instead of generated per trial."
        ),
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Run models sequentially in a single process (default full mode uses sharded windows).",
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
            run_mode=str(manifest["run_mode"]),
        )
        _write_json(_worker_done_path(args.run_dir, args.model), stats)
        print(f"Worker done: {args.model} -> D_nonempty={stats['d_nonempty']}/{stats['n']} errors={stats['errors']}")
        return

    shard = (args.mode == "full") and (not args.serial)
    run_experiment(run_mode=args.mode, seed=args.seed, shard=shard, fixed_b_csv=args.fixed_b_csv)


if __name__ == "__main__":
    main()
