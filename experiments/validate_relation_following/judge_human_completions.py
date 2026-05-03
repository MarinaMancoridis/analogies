"""
LLM panel judgments on eligible human completions (relation grade = same pipeline
as run_relation_triples: relation module ``_majority_judge`` with ``DEFAULT_JUDGE_N=1``).

**Smoke** (10 completions per shard; three parallel workers)::

  cd /path/to/analogies
  ./experiments/validate_relation_following/run_human_judgment_panel.sh smoke

Or single shard::

  python experiments/validate_relation_following/judge_human_completions.py --smoke \\
    --shard-model openai/gpt-5.4-mini \\
    --out-ndjson experiments/validate_relation_following/human_judgments_panel_smoke.ndjson

**Full** (~1300 completions per shard, three judge models in parallel + tqdm bars)::

  pip install tqdm   # once, for progress bars
  ./experiments/validate_relation_following/run_human_judgment_panel.sh full

Each NDJSON line includes ``completion_prompts`` (all ``PROMPT_VARIATIONS``),
``completion_prompt_colon``, ``relation_text``, ``grade_prompt``, ``judge_raws``, etc.

Resume: re-run the same command; already-written (judge_model, completion_id) pairs are skipped.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None  # type: ignore[misc, assignment]

# Panel shards (each model processes the full eligible set; run three processes in parallel).
SHARD_MODELS: Tuple[str, ...] = (
    "openai/gpt-5.4-mini",
    "google/gemini-3.1-flash-lite-preview",
    "deepseek/deepseek-v3.2",
)

SMOKE_LIMIT_DEFAULT = 10


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _load_run_relation_triples():
    path = (
        Path(__file__).resolve().parent.parent
        / "LLM_relations_completions"
        / "run_relation_triples.py"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("run_relation_triples", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    # run_relation_triples expects imports from repo; ensure repo root on path.
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # Required so @dataclass and similar can resolve cls.__module__ during exec_module.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _append_ndjson_locked(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = str(path) + ".lock"
    with open(lock_path, "w", encoding="utf-8") as lockf:
        import fcntl

        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        finally:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)


def _load_done_keys(path: Path) -> Set[Tuple[str, str]]:
    """Pairs (judge_model, completion_id) already written."""
    if not path.is_file():
        return set()
    done: Set[Tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("record_type") != "human_completion_judgment":
                continue
            jm = rec.get("judge_model")
            cid = rec.get("completion_id")
            if isinstance(jm, str) and isinstance(cid, str):
                done.add((jm, cid))
    return done


def _utc_now() -> str:
    return dt.datetime.utcnow().isoformat() + "Z"


def _tqdm_desc(shard_model: str) -> str:
    tail = shard_model.split("/")[-1]
    return tail[:28] if len(tail) > 28 else tail


def _maybe_tqdm(
    items: List[Dict[str, Any]],
    *,
    shard_model: str,
    use_bar: bool,
    position: Optional[int],
) -> Iterable[Dict[str, Any]]:
    if not use_bar or _tqdm is None or not items:
        return items
    kwargs: Dict[str, Any] = {
        "total": len(items),
        "desc": _tqdm_desc(shard_model),
        "unit": "judg",
        "file": sys.stderr,
        "mininterval": 0.3,
        "dynamic_ncols": True,
        "leave": True,
    }
    if position is not None:
        kwargs["position"] = position
    return _tqdm(items, **kwargs)


def _completion_prompts_all(module: Any, a: str, b: str, c: str) -> Dict[str, str]:
    """All PROMPT_VARIATIONS strings (same task as LLM D-completion), keyed by prompt_type."""
    out: Dict[str, str] = {}
    pv = getattr(module, "PROMPT_VARIATIONS", None)
    if not isinstance(pv, dict):
        return out
    for key, fn in sorted(pv.items()):
        if not callable(fn):
            continue
        try:
            out[str(key)] = fn(a, b, c)
        except Exception:
            out[str(key)] = ""
    return out


def _grade_prompt_fallback(module: Any, a: str, b: str, c: str, d: str) -> str:
    fn = getattr(module, "_prompt_grade", None)
    if callable(fn):
        try:
            return fn(a, b, c, d)
        except Exception:
            return ""
    return ""


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description=(
            "One shard: run relation-following judgments for eligible human completions "
            "using a single judge model. Use --smoke for a short run; run three processes "
            "(one per --shard-model) for the full panel."
        )
    )
    p.add_argument(
        "--in-json",
        type=Path,
        default=here / "human_completions_flat.json",
        help="Output of validate_human_responses.py.",
    )
    p.add_argument(
        "--out-ndjson",
        type=Path,
        default=here / "human_judgments_panel.ndjson",
        help="Shared NDJSON output (append + file lock; safe across parallel shards).",
    )
    p.add_argument(
        "--shard-model",
        type=str,
        required=True,
        choices=SHARD_MODELS,
        help="Judge model for this worker (one shard per model).",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help=f"Process at most {SMOKE_LIMIT_DEFAULT} completions (after resume skip).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max completions to process this run (after resume skip). Overrides --smoke when set.",
    )
    p.add_argument(
        "--delay-seconds",
        type=float,
        default=0.0,
        help="Optional pause after each API call.",
    )
    p.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable tqdm progress bar.",
    )
    p.add_argument(
        "--tqdm-position",
        type=int,
        default=None,
        metavar="N",
        help=(
            "tqdm line index when running several shards in one terminal (e.g. 0, 1, 2). "
            "The panel script sets this automatically."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    limit = args.limit if args.limit is not None else (SMOKE_LIMIT_DEFAULT if args.smoke else None)

    rrt = _load_run_relation_triples()

    payload = json.loads(args.in_json.read_text(encoding="utf-8"))
    all_rows: List[Dict[str, Any]] = payload["completions"]
    eligible = [r for r in all_rows if r.get("participant_eligible_complete")]
    done = _load_done_keys(args.out_ndjson)
    pending: List[Dict[str, Any]] = []
    for r in eligible:
        k = (args.shard_model, str(r["completion_id"]))
        if k in done:
            continue
        pending.append(r)
        if limit is not None and len(pending) >= limit:
            break

    use_bar = not args.no_progress_bar and _tqdm is not None
    if not use_bar and not args.no_progress_bar and _tqdm is None:
        print(
            "Note: install tqdm for a progress bar: pip install tqdm",
            file=sys.stderr,
            flush=True,
        )

    print(
        f"Shard {args.shard_model}: {len(eligible)} eligible in JSON, "
        f"{len(done)} judgments already in output for this model, "
        f"{len(pending)} to process this run"
        + (f" (limit={limit})" if limit is not None else "")
        + ".",
        flush=True,
    )

    processed = 0
    iterator = _maybe_tqdm(
        pending,
        shard_model=args.shard_model,
        use_bar=use_bar,
        position=args.tqdm_position,
    )
    for r in iterator:
        relation_key = str(r["relation_key"])
        a, b, c, d = str(r["A"]), str(r["B"]), str(r["C"]), str(r["D"])
        module = rrt._load_relation_module(relation_key)

        completion_prompts = _completion_prompts_all(module, a, b, c)
        prompt_colon = completion_prompts.get("colon", "")

        relation_text = str(getattr(module, "REL_TEXT", "") or "")
        relation_name_module = str(getattr(module, "REL_NAME", "") or "")

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
        err: str | None = None
        try:
            judge = rrt._run_relation_judge(module, args.shard_model, a, b, c, d)
        except Exception as e:
            err = str(e)

        grade_prompt = judge["judge_prompt"] or _grade_prompt_fallback(module, a, b, c, d)

        rec: Dict[str, Any] = {
            "record_type": "human_completion_judgment",
            "timestamp_utc": _utc_now(),
            "judge_model": args.shard_model,
            "completion_id": r["completion_id"],
            "response_id": r["response_id"],
            "prolific_pid": r.get("prolific_pid", ""),
            "set_number": r["set_number"],
            "item_number": r["item_number"],
            "triple_id": r["triple_id"],
            "relation_key": relation_key,
            "relation_name": r.get("relation_name", "") or relation_name_module,
            "relation_text": relation_text,
            "A": a,
            "B": b,
            "C": c,
            "D_human": d,
            "completion_prompt_colon": prompt_colon,
            "completion_prompts": completion_prompts,
            "grade_prompt": grade_prompt,
            "grade_raw": judge["judge_raws"][-1] if judge["judge_raws"] else "",
            "judge_raws": judge["judge_raws"],
            "judge_labels": judge["judge_labels"],
            "judge_counts": judge["judge_counts"],
            "judge_majority_label": judge["judge_majority_label"],
            "judge_majority_correct": judge["judge_majority_correct"],
            "judge_n": judge["judge_n"],
            "judge_error": judge["judge_error"],
            "error": err,
        }
        _append_ndjson_locked(args.out_ndjson, rec)
        processed += 1
        if args.delay_seconds > 0:
            time.sleep(args.delay_seconds)

    print(f"Shard {args.shard_model}: wrote {processed} new lines to {args.out_ndjson}", flush=True)


if __name__ == "__main__":
    main()
