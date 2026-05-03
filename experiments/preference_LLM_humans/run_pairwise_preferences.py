from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

JUDGE_MODELS: List[str] = [
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


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Create predetermined human-vs-LLM pairwise preference questions and "
            "run LLM judges over random subsets."
        )
    )
    parser.add_argument(
        "--gold-json",
        type=Path,
        default=here / "gold_standard_humans_models.json",
        help="Linked gold JSON with triples/human/model slots.",
    )
    parser.add_argument(
        "--predetermined-json",
        type=Path,
        default=here / "predetermined_pairwise_questions.json",
        help="Where to write/read predetermined pairwise questions.",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=here / "pairwise_llm_judgments.ndjson",
        help="Single NDJSON output file; judgments stream in as they complete.",
    )
    parser.add_argument(
        "--build-pool-size",
        type=int,
        default=1000,
        help="Number of predetermined comparisons to create.",
    )
    parser.add_argument(
        "--comparisons-per-model",
        type=int,
        default=100,
        help="How many predetermined comparisons each judge model evaluates.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="RNG seed for deterministic question construction/sampling.",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Only build predetermined question JSON, do not call LLM judges.",
    )
    parser.add_argument(
        "--use-existing-predetermined",
        action="store_true",
        help="Use existing predetermined JSON instead of rebuilding it.",
    )
    parser.add_argument(
        "--required-llm-prompt-type",
        type=str,
        default="",
        help="Optional filter for llm_source.prompt_type (e.g., colon).",
    )
    return parser.parse_args()


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _norm(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _extract_choice(raw: str) -> Optional[int]:
    txt = (raw or "").strip()
    m = re.search(r"(?:ANSWER|CHOICE)\s*:\s*([12])", txt, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m2 = re.search(r"\b([12])\b", txt)
    if m2:
        return int(m2.group(1))
    return None


def _build_question_prompt(item: Dict[str, Any]) -> str:
    return (
        "Choose which completion better fits the analogy.\n"
        "Output ONLY in this format:\n"
        "ANSWER: 1 or 2\n\n"
        f"Analogy: {item['A']} : {item['B']} :: {item['C']} : ____\n"
        f"Completion 1: {item['completion_1_text']}\n"
        f"Completion 2: {item['completion_2_text']}\n"
    )


def _progress_bar(i: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[no-work]"
    filled = int(width * i / total)
    bar = "#" * filled + "-" * (width - filled)
    pct = 100.0 * i / total
    return f"[{bar}] {i}/{total} ({pct:5.1f}%)"


def _filter_predetermined_questions(
    predetermined: Dict[str, Any], required_llm_prompt_type: str
) -> Dict[str, Any]:
    required = (required_llm_prompt_type or "").strip()
    if not required:
        return predetermined
    questions = predetermined.get("questions", [])
    filtered = [
        q
        for q in questions
        if q.get("llm_source", {}).get("prompt_type") == required
    ]
    out = dict(predetermined)
    out["questions"] = filtered
    out["llm_prompt_type_filter"] = required
    return out


def _collect_eligible_pairs(gold_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for t in gold_payload.get("triples", []):
        triple_id = int(t["triple_id"])
        relation_key = t["relation_key"]
        a, b, c = t["A"], t["B"], t["C"]

        human_candidates = [
            h
            for h in t.get("human_slots", [])
            if h.get("participant_id") and _norm(h.get("response")) != ""
        ]
        if not human_candidates:
            continue

        llm_candidates: List[Dict[str, Any]] = []
        for ms in t.get("model_slots", []):
            model_name = ms["model_name"]
            for prompt_type, resp in ms.get("responses_by_prompt_type", {}).items():
                d = _norm(resp.get("D"))
                if d == "":
                    continue
                llm_candidates.append(
                    {
                        "model_name": model_name,
                        "prompt_type": prompt_type,
                        "D": resp.get("D"),
                        "grade_correct": resp.get("grade_correct"),
                        "error": resp.get("error"),
                    }
                )
        if not llm_candidates:
            continue

        pairs.append(
            {
                "triple_id": triple_id,
                "relation_key": relation_key,
                "A": a,
                "B": b,
                "C": c,
                "human_candidates": human_candidates,
                "llm_candidates": llm_candidates,
            }
        )
    return pairs


def build_predetermined_questions(
    *,
    gold_payload: Dict[str, Any],
    pool_size: int,
    seed: int,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    eligible = _collect_eligible_pairs(gold_payload)
    if not eligible:
        raise ValueError("No eligible triples with both human and LLM completions.")

    questions: List[Dict[str, Any]] = []
    for qid in range(1, pool_size + 1):
        base = rng.choice(eligible)
        human_pick = rng.choice(base["human_candidates"])
        llm_pick = rng.choice(base["llm_candidates"])

        human_text = str(human_pick["response"]).strip()
        llm_text = str(llm_pick["D"]).strip()
        # Randomize which side gets human vs llm.
        human_on_left = bool(rng.getrandbits(1))
        completion_1_text = human_text if human_on_left else llm_text
        completion_2_text = llm_text if human_on_left else human_text

        question = {
            "comparison_id": qid,
            "triple_id": base["triple_id"],
            "relation_key": base["relation_key"],
            "A": base["A"],
            "B": base["B"],
            "C": base["C"],
            "human_source": {
                "participant_id": human_pick["participant_id"],
                "prolific_pid": human_pick.get("prolific_pid"),
                "set_number": human_pick.get("set_number"),
                "item_number": human_pick.get("item_number"),
                "response": human_text,
                "confidence": human_pick.get("confidence"),
            },
            "llm_source": {
                "model_name": llm_pick["model_name"],
                "prompt_type": llm_pick["prompt_type"],
                "response": llm_text,
                "grade_correct": llm_pick.get("grade_correct"),
                "error": llm_pick.get("error"),
            },
            "completion_1_text": completion_1_text,
            "completion_2_text": completion_2_text,
            "completion_1_source": "human" if human_on_left else "llm",
            "completion_2_source": "llm" if human_on_left else "human",
        }
        questions.append(question)

    return {
        "generated_at_utc": _utc_now(),
        "seed": seed,
        "pool_size": pool_size,
        "source_gold_json": "gold_standard_humans_models.json",
        "questions": questions,
    }


def run_judges(
    *,
    predetermined: Dict[str, Any],
    comparisons_per_model: int,
    seed: int,
    results_path: Path,
) -> Dict[str, Any]:
    # Lazy import so --skip-judge mode does not require API client deps.
    import sys

    _HERE = Path(__file__).resolve().parent
    _REPO_ROOT = _HERE.parent.parent
    if str(_REPO_ROOT.parent) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT.parent))
    from analogies.utils import generate_inference

    rng = random.Random(seed + 1)
    questions = predetermined["questions"]
    if len(questions) < comparisons_per_model:
        raise ValueError(
            f"Need at least {comparisons_per_model} predetermined questions; got {len(questions)}"
        )

    summary_by_model: Dict[str, Dict[str, Any]] = {}
    summary_lock = threading.Lock()
    write_lock = threading.Lock()

    # Start a fresh output file each run.
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text("", encoding="utf-8")
    with write_lock:
        with results_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "record_type": "run_meta",
                        "generated_at_utc": _utc_now(),
                        "comparisons_per_model": comparisons_per_model,
                        "judge_models": JUDGE_MODELS,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    sampled_by_model: Dict[str, List[Dict[str, Any]]] = {
        model: rng.sample(questions, comparisons_per_model) for model in JUDGE_MODELS
    }

    def _run_one_model(model: str) -> Tuple[str, Dict[str, Any]]:
        sampled = sampled_by_model[model]
        human_chosen = 0
        llm_chosen = 0
        invalid = 0

        print(f"\n[{model}] judging {comparisons_per_model} pairwise comparisons...")
        for i, q in enumerate(sampled, start=1):
            prompt = _build_question_prompt(q)
            raw = ""
            choice: Optional[int] = None
            err: Optional[str] = None
            try:
                raw = generate_inference(prompt, model)
                choice = _extract_choice(raw)
            except Exception as e:  # keep moving
                err = str(e)

            picked_source: Optional[str] = None
            if choice == 1:
                picked_source = q["completion_1_source"]
            elif choice == 2:
                picked_source = q["completion_2_source"]
            else:
                invalid += 1

            if picked_source == "human":
                human_chosen += 1
            elif picked_source == "llm":
                llm_chosen += 1

            record = {
                "record_type": "judgment",
                "judge_model": model,
                "comparison_id": q["comparison_id"],
                "triple_id": q["triple_id"],
                "choice": choice,
                "picked_source": picked_source,
                "raw_judge_response": raw,
                "error": err,
            }
            with write_lock:
                with results_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(
                f"  [{model}] {_progress_bar(i, comparisons_per_model)} | "
                f"comparison_id={q['comparison_id']} choice={choice} "
                f"picked={picked_source}"
            )

        valid = human_chosen + llm_chosen
        summary = {
            "n_total": comparisons_per_model,
            "n_valid": valid,
            "n_invalid": invalid,
            "human_chosen": human_chosen,
            "llm_chosen": llm_chosen,
            "pct_human_chosen_over_total": (human_chosen / comparisons_per_model) if comparisons_per_model else 0.0,
            "pct_human_chosen_over_valid": (human_chosen / valid) if valid else 0.0,
        }
        with summary_lock:
            summary_by_model[model] = summary
        with write_lock:
            with results_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "record_type": "model_summary",
                            "judge_model": model,
                            **summary,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return model, summary

    # Run one shard per evaluator model for speed.
    with ThreadPoolExecutor(max_workers=len(JUDGE_MODELS)) as ex:
        list(ex.map(_run_one_model, JUDGE_MODELS))

    with write_lock:
        with results_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "record_type": "run_summary",
                        "generated_at_utc": _utc_now(),
                        "comparisons_per_model": comparisons_per_model,
                        "judge_models": JUDGE_MODELS,
                        "summary_by_model": summary_by_model,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return {
        "generated_at_utc": _utc_now(),
        "comparisons_per_model": comparisons_per_model,
        "judge_models": JUDGE_MODELS,
        "summary_by_model": summary_by_model,
    }


def main() -> None:
    args = parse_args()
    if args.use_existing_predetermined:
        predetermined = json.loads(args.predetermined_json.read_text(encoding="utf-8"))
        print(f"Loaded predetermined question pool: {args.predetermined_json}")
    else:
        gold_payload = json.loads(args.gold_json.read_text(encoding="utf-8"))
        predetermined = build_predetermined_questions(
            gold_payload=gold_payload,
            pool_size=args.build_pool_size,
            seed=args.seed,
        )
        args.predetermined_json.write_text(
            json.dumps(predetermined, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Wrote predetermined question pool: {args.predetermined_json}")

    predetermined = _filter_predetermined_questions(
        predetermined, args.required_llm_prompt_type
    )
    if args.required_llm_prompt_type:
        print(
            f"Applied llm_source.prompt_type filter='{args.required_llm_prompt_type}': "
            f"{len(predetermined['questions'])} questions remain"
        )

    if args.skip_judge:
        print("Skipping judge calls (--skip-judge).")
        return

    judged = run_judges(
        predetermined=predetermined,
        comparisons_per_model=args.comparisons_per_model,
        seed=args.seed,
        results_path=args.results_json,
    )
    print(f"\nWrote streamed judge output: {args.results_json}")
    print("\n=== Human-vs-LLM preference summary ===")
    for model, s in judged["summary_by_model"].items():
        print(
            f"{model}: human_chosen={s['human_chosen']}/{s['n_total']} "
            f"({100*s['pct_human_chosen_over_total']:.1f}% total; "
            f"{100*s['pct_human_chosen_over_valid']:.1f}% valid)"
        )


if __name__ == "__main__":
    main()
