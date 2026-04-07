"""
Run analogies across many models for any task module.

UPDATED DESIGN:
- Works with any task module implementing:
    - generate_dataset(n, rng) -> (list_of_tuples, stats)
    - run_trial(model, *words, prompt_type) -> trial dict
- Stimuli are sampled ONCE using task-specific generator.
- All models evaluate the SAME tuples.
- Words are sampled from common_mod.ALL.
"""

from __future__ import annotations

import os
import json
import uuid
import random
import datetime
import warnings
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import nltk
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency

# -------------------------------
# Suppress unwanted warnings
# -------------------------------
warnings.filterwarnings("ignore", category=FutureWarning)

from analogies import common as common_mod

# Different types of tasks to run
# from analogies.analogy_types.structural import identity as task_mod # classic identity task (good!!! 1)
# from analogies.analogy_types.structural import one_cycle as task_mod # 1-cycle analogy task (bad)
# from analogies.analogy_types.structural import adjacent_repetition as task_mod # adjacent repetition task (bad)
# from analogies.analogy_types.structural import reverse_copy as task_mod # reverse copy task (bad)
# from analogies.analogy_types.structural import rotation as task_mod # rotation task (bad)
# from analogies.analogy_types.structural import duplicate_first as task_mod # duplicate first task (bad)
# from analogies.analogy_types.structural import delete_middle as task_mod # delete middle task (bad)
# from analogies.analogy_types.structural import endpoints_only as task_mod # endpoints only task (bad)
# from analogies.analogy_types.structural import repeat_count as task_mod # repeat count task (maybe, bc it only worked on wishy washy. try this exclusively with first half words next?)
# from analogies.analogy_types.structural import fixed_offset_position as task_mod # fixed offset position task (bad)
# from analogies.analogy_types.structural import unbracket as task_mod # unbracket task (bad)
# from analogies.analogy_types.structural import uppercase_transform as task_mod # uppercase transformation task (bad)
# from analogies.analogy_types.relational.class_inclusion import class_inclusion as task_mod  # (good!!! 2)
# from analogies.analogy_types.relational.part_whole import part_whole as task_mod # (good!!! 3)
# from analogies.analogy_types.relational.similar import similar as task_mod # (meh but keep 4)
# from analogies.analogy_types.relational.contrast import contrast as task_mod # 
# from analogies.analogy_types.relational.attribute import attribute as task_mod # 
# from analogies.analogy_types.relational.non_attribute import non_attribute as task_mod # 
# from analogies.analogy_types.relational.case_relations import case_relations as task_mod # 
# from analogies.analogy_types.relational.cause_purpose import cause_purpose as task_mod 
from analogies.analogy_types.relational.space_time import space_time as task_mod 


# -------------------------------
# CONFIG
# -------------------------------
MODELS = [
    # "gpt-5",
    # "gpt-5-pro",
    "gpt-5.2",
    # "o3",
    # "o1",
    # "google/gemini-3.1-pro-preview",
    "google/gemini-3-flash-preview",
    # "claude-3-5-sonnet-20241022",
    "anthropic/claude-opus-4.5",
    # "x-ai/grok-4"
]

N_TRIALS_PER_MODEL = 3
SEED = 12345

BRYS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "concreteness.txt")
HERE = os.path.dirname(__file__)
RUNS_DIR = os.path.join(HERE, "runs")

# -------------------------------
# Utilities
# -------------------------------
def _make_run_dir(base: str) -> str:
    run_id = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    out = os.path.join(base, run_id)
    os.makedirs(out, exist_ok=True)
    return out


def _append_ndjson(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _utc_now() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


# -------------------------------
# Lexical Metrics
# -------------------------------
_RANK_INDEX: Optional[Dict[str, int]] = None


def _ensure_rank_index() -> None:
    global _RANK_INDEX
    if _RANK_INDEX is not None:
        return
    _RANK_INDEX = {w.lower(): i + 1 for i, w in enumerate(common_mod.ALL)}


def _pos_universal(word: str) -> Optional[str]:
    try:
        tag = nltk.pos_tag([word], tagset="universal")[0][1]
        return tag
    except Exception:
        return None


def _polysemy_count(word: str) -> int:
    try:
        return len(wn.synsets(word.lower()))
    except Exception:
        return 0


def _brys_score(word: str) -> Optional[float]:
    try:
        return common_mod._brys_score(word)
    except Exception:
        return None


def _brys_label(word: str) -> Optional[str]:
    try:
        return common_mod.ac_label_brys(word)
    except Exception:
        return None


def word_metrics(word: str) -> Dict[str, Any]:
    w = word.lower().strip()
    _ensure_rank_index()
    return {
        "len": len(w),
        "pop_rank": _RANK_INDEX.get(w),
        "pop_zipf": float(zipf_frequency(w, "en")),
        "polysemy": _polysemy_count(w),
        "brys": _brys_score(w),
        "brys_label": _brys_label(w),
        "pos": _pos_universal(w),
    }


def _valid_word(word: str) -> bool:
    try:
        m = word_metrics(word)
    except Exception:
        return False
    required_fields = ["pos", "pop_zipf", "pop_rank"]
    return all(m.get(f) is not None for f in required_fields)


# -------------------------------
# Trial evaluation dataclass
# -------------------------------
@dataclass
class Counts:
    n: int = 0
    hits: int = 0


# -------------------------------
# Dynamic trial evaluation
# -------------------------------
def evaluate_trial(model: str, words: tuple, prompt_types: Optional[List[str]] = None) -> list[dict]:
    """
    Evaluate a tuple of arbitrary length words for one or more prompt types.
    """
    if prompt_types is None:
        prompt_types = list(getattr(task_mod, "PROMPT_VARIATIONS", {}).keys())

    trials = []
    for ptype in prompt_types:
        trial = task_mod.run_trial(model, *words, prompt_type=ptype)
        trials.append(trial)

    return trials


# -------------------------------
# Main
# -------------------------------
def main() -> None:
    rng = random.Random(SEED)

    try:
        common_mod.load_brysbaert_norms(BRYS_PATH)
    except Exception:
        pass

    out_dir = _make_run_dir(RUNS_DIR)
    print(f"Run dir: {out_dir}")

    # -------------------------------
    # Generate dataset ONCE
    # -------------------------------
    dataset_generator = getattr(task_mod, "generate_dataset")
    dataset, sampling_stats = dataset_generator(N_TRIALS_PER_MODEL, rng)
    print(f"Generated {len(dataset)} fixed stimuli tuples.")

    # Subset for prompt-variation robustness
    n_subset = min(30, len(dataset))
    prompt_robustness_subset = rng.sample(dataset, n_subset)
    prompt_types_to_test = list(getattr(task_mod, "PROMPT_VARIATIONS", {}).keys())

    run_summary = {
        "run_id": os.path.basename(out_dir),
        "seed": SEED,
        "n_trials": N_TRIALS_PER_MODEL,
        "models": {},
        "stimuli": dataset,
        "sampling_spec": sampling_stats,
    }

    grand_total = Counts()

    for model in MODELS:
        model_dir = os.path.join(out_dir, model.replace("/", "_"))
        os.makedirs(model_dir, exist_ok=True)
        trials_path = os.path.join(model_dir, "trials.ndjson")

        overall_counts = Counts()
        prompt_type_counts: Dict[str, Counts] = {ptype: Counts() for ptype in prompt_types_to_test}

        # 1) Default prompt "colon" for all tuples
        for tup in dataset:
            trial = evaluate_trial(model, tup, prompt_types=["colon"])[0]
            _append_ndjson(trials_path, trial)

            hit = int(trial.get("is_identity", False))

            overall_counts.n += 1
            overall_counts.hits += hit
            prompt_type_counts["colon"].n += 1
            prompt_type_counts["colon"].hits += hit

        # 2) Prompt-variation robustness on subset
        for tup in prompt_robustness_subset:
            for trial in evaluate_trial(model, tup, prompt_types=prompt_types_to_test):
                if trial["prompt_type"] == "colon":
                    continue

                _append_ndjson(trials_path, trial)

                hit = int(trial.get("is_identity", False))

                overall_counts.n += 1
                overall_counts.hits += hit
                prompt_type_counts[trial["prompt_type"]].n += 1
                prompt_type_counts[trial["prompt_type"]].hits += hit

        model_success_rate = overall_counts.hits / overall_counts.n if overall_counts.n else 0.0

        run_summary["models"][model] = {
            "n": overall_counts.n,
            "hits": overall_counts.hits,
            "success_rate": model_success_rate,
            "success_fraction": f"{overall_counts.hits}/{overall_counts.n}",
            "prompt_types": {
                ptype: {
                    "n": counts.n,
                    "hits": counts.hits,
                    "success_rate": counts.hits / counts.n if counts.n else 0.0,
                    "success_fraction": f"{counts.hits}/{counts.n}",
                }
                for ptype, counts in prompt_type_counts.items()
            },
        }

        grand_total.n += overall_counts.n
        grand_total.hits += overall_counts.hits

        print(f"[{model}] hit_rate={model_success_rate:.3f} ({overall_counts.hits}/{overall_counts.n})")

    grand_success_rate = grand_total.hits / grand_total.n if grand_total.n else 0.0
    run_summary["overall"] = {
        "n": grand_total.n,
        "hits": grand_total.hits,
        "success_rate": grand_success_rate,
        "success_fraction": f"{grand_total.hits}/{grand_total.n}",
    }

    _write_json(os.path.join(out_dir, "run_summary.json"), run_summary)

    print("\n=== DONE ===")
    print(f"OVERALL SUCCESS RATE: {grand_total.hits}/{grand_total.n} = {grand_success_rate:.3f}")
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()