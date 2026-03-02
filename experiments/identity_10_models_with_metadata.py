"""
Run identity analogies A:A::C:_ across many models.

UPDATED DESIGN:
- Stimuli (A,C pairs) are sampled ONCE.
- All models evaluate the SAME word pairs.
- Words are sampled from common_mod.ALL.
- We retry sampling until all lexical metadata is available.

Outputs under:
  analogies/experiments/runs/<run_id>/<model>/trials.ndjson
"""

from __future__ import annotations

import os
import json
import uuid
import random
import datetime
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import nltk
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency

from analogies.analogy_types import identity as identity_mod
from analogies import common as common_mod


# -------------------------------
# CONFIG
# -------------------------------

MODELS = [
    "gpt-5",
    "gpt-5-pro",
    "gpt-5.2",
    "o3",
    "o1",
    "google/gemini-3.1-pro-preview",
    "gemini-2.0-flash-exp",
    "claude-3-5-sonnet-20241022",
    "anthropic/claude-opus-4.5",
    "x-ai/grok-4"
]

N_TRIALS_PER_MODEL = 1
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


# -------------------------------
# ### NEW: DATASET GENERATION
# -------------------------------

def _valid_word(word: str) -> bool:
    """
    ### NEW
    Accept only words where ALL required metadata is available.
    Retry if any field is missing.
    """
    try:
        m = word_metrics(word)
    except Exception:
        return False

    if m["pos"] is None:
        return False
    if m["pop_zipf"] is None:
        return False
    if m["pop_rank"] is None:
        return False

    return True

def generate_identity_dataset(n: int, rng: random.Random):
    """
    Sample (A,C) pairs ONCE from common_mod.ALL.
    Retry until both words pass metadata validation.
    Returns:
        dataset: List[(A,C)]
        stats: dict with sampling metadata
    """
    words = [w for w in common_mod.ALL if isinstance(w, str)]
    initial_pool_size = len(words)

    dataset = []
    rejected = 0
    attempts = 0

    while len(dataset) < n:
        attempts += 1

        A = rng.choice(words)
        C = rng.choice(words)

        if A == C:
            rejected += 1
            continue

        if not _valid_word(A):
            rejected += 1
            continue

        if not _valid_word(C):
            rejected += 1
            continue

        dataset.append((A, C))

    sampling_stats = {
        "source_vocab": "common_mod.ALL",
        "initial_pool_size": initial_pool_size,
        "sampling_with_replacement": True,
        "pair_sampling_independent": True,
        "requires_metadata_fields": [
            "pos",
            "pop_zipf",
            "pop_rank",
        ],
        "total_attempts": attempts,
        "total_rejected": rejected,
        "acceptance_rate": len(dataset) / attempts if attempts else None,
    }

    return dataset, sampling_stats


# -------------------------------
# Trial evaluation
# -------------------------------

@dataclass
class Counts:
    n: int = 0
    hits: int = 0

def infer_hit(trial: Dict[str, Any]) -> bool:
    return bool(trial.get("is_identity", False))


def evaluate_trial(model: str, A: str, C: str, prompt_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Evaluate fixed (A,C) pair for one or more prompt variations.
    Returns a list of trial dicts (one per prompt_type).
    """
    if prompt_types is None:
        prompt_types = ["colon"]  # default if not specified

    trials = []

    for ptype in prompt_types:
        prompt_fn = identity_mod.PROMPT_VARIATIONS.get(ptype, identity_mod._prompt)
        prompt = prompt_fn(A, C) if callable(prompt_fn) else identity_mod._prompt(A, C)

        raw = identity_mod.generate_inference(prompt, model)
        pred = identity_mod.clean_answer(raw)
        hit = (pred == C.lower())

        trial = {
            "model": model,
            "analogy_type": "identity",
            "A": A,
            "C": C,
            "prompt": prompt,
            "raw_response": raw,
            "parsed_answer": pred,
            "expected": C,
            "is_identity": hit,
            "prompt_type": ptype
        }

        # attach lexical metadata
        A_m = word_metrics(A)
        C_m = word_metrics(C)
        for k, v in A_m.items():
            trial[f"A_{k}"] = v
        for k, v in C_m.items():
            trial[f"C_{k}"] = v

        trial["experiment"] = "identity_uniform_all_words"

        trials.append(trial)

    return trials


# -------------------------------
# Main
# -------------------------------

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
    dataset, sampling_stats = generate_identity_dataset(N_TRIALS_PER_MODEL, rng)
    print(f"Generated {len(dataset)} fixed stimuli pairs.")

    # Select a subset of 30 pairs (or all if smaller) for prompt-variation robustness
    n_subset = min(30, len(dataset))
    prompt_robustness_subset = rng.sample(dataset, n_subset)
    prompt_types_to_test = ["colon", "english", "one_shot", "few_shot"]

    run_summary = {
        "run_id": os.path.basename(out_dir),
        "seed": SEED,
        "n_trials": N_TRIALS_PER_MODEL,
        "models": {},
        "stimuli": dataset,
        "sampling_spec": sampling_stats,
    }

    for model in MODELS:
        model_dir = os.path.join(out_dir, model.replace("/", "_"))
        os.makedirs(model_dir, exist_ok=True)
        trials_path = os.path.join(model_dir, "trials.ndjson")

        # Counts for overall + per-prompt
        overall_counts = Counts()
        prompt_type_counts: Dict[str, Counts] = {ptype: Counts() for ptype in prompt_types_to_test}

        # 1️⃣ Run normal evaluation (default prompt "colon") for all pairs
        for (A, C) in dataset:
            trial = evaluate_trial(model, A, C, prompt_types=["colon"])[0]
            _append_ndjson(trials_path, trial)

            overall_counts.n += 1
            overall_counts.hits += int(infer_hit(trial))
            prompt_type_counts["colon"].n += 1
            prompt_type_counts["colon"].hits += int(infer_hit(trial))

        # 2️⃣ Run prompt-variation robustness on subset (skip "colon" duplicates)
        for (A, C) in prompt_robustness_subset:
            for trial in evaluate_trial(model, A, C, prompt_types=prompt_types_to_test):
                if trial["prompt_type"] == "colon":
                    continue  # already ran
                _append_ndjson(trials_path, trial)

                overall_counts.n += 1
                overall_counts.hits += int(infer_hit(trial))
                prompt_type_counts[trial["prompt_type"]].n += 1
                prompt_type_counts[trial["prompt_type"]].hits += int(infer_hit(trial))

        # Save summary for this model
        run_summary["models"][model] = {
            "n": overall_counts.n,
            "hits": overall_counts.hits,
            "success_rate": overall_counts.hits / overall_counts.n if overall_counts.n else 0.0,
            "prompt_types": {
                ptype: {
                    "n": counts.n,
                    "hits": counts.hits,
                    "success_rate": counts.hits / counts.n if counts.n else 0.0
                } for ptype, counts in prompt_type_counts.items()
            }
        }

        print(f"[{model}] hit_rate={run_summary['models'][model]['success_rate']:.3f}")

    _write_json(os.path.join(out_dir, "run_summary.json"), run_summary)

    print("\n=== DONE ===")
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()