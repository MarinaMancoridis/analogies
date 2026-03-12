from analogies.utils import generate_inference
from analogies.common import clean_answer, ALL, _brys_score, ac_label_brys
from typing import Dict, Optional
import random
import nltk
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency

# -------------------------------
# Task name
# -------------------------------
TASK_NAME = "unbracket"

# -------------------------------
# Prompt variations
# Pattern: (A) : A :: (B) : ____
# Rule: remove parentheses
# Expected answer: B
# -------------------------------
PROMPT_VARIATIONS = {
    "colon": lambda A, B: (
        f"Complete the analogy. Reply ONLY as: ANSWER: <word>\n\n"
        f"({A}) : {A} :: ({B}) : ____"
    ),
    "english": lambda A, B: (
        f"Complete the pattern. Reply ONLY as: ANSWER: <word>\n\n"
        f"The second item removes the parentheses from the first item.\n"
        f"({A}) : {A} :: ({B}) : ____"
    ),
    "one_shot": lambda A, B: (
        f"Complete the analogy. Reply ONLY as: ANSWER: <word>\n\n"
        f"({A}) : {A} :: ({B}) : ____\n\n"
        f"Here is one example:\n"
        f"(cat) : cat :: (tree) : tree"
    ),
    "few_shot": lambda A, B: (
        f"Complete the analogy. Reply ONLY as: ANSWER: <word>\n\n"
        f"({A}) : {A} :: ({B}) : ____\n\n"
        f"Here are some examples:\n"
        f"(cat) : cat :: (tree) : tree\n"
        f"(cup) : cup :: (bird) : bird"
    ),
}

# -------------------------------
# Word metrics
# -------------------------------
_RANK_INDEX: Optional[Dict[str, int]] = None

def _ensure_rank_index() -> None:
    global _RANK_INDEX
    if _RANK_INDEX is not None:
        return
    _RANK_INDEX = {w.lower(): i + 1 for i, w in enumerate(ALL)}

def word_metrics(word: str) -> Dict[str, Optional[float]]:
    w = word.lower().strip()
    _ensure_rank_index()
    try:
        pos_tag = nltk.pos_tag([w], tagset="universal")[0][1]
    except Exception:
        pos_tag = None
    try:
        poly = len(wn.synsets(w))
    except Exception:
        poly = None
    try:
        brys = _brys_score(w)
        brys_label = ac_label_brys(w)
    except Exception:
        brys = brys_label = None
    try:
        zipf = float(zipf_frequency(w, "en"))
    except Exception:
        zipf = None

    return {
        "len": len(w),
        "pop_rank": _RANK_INDEX.get(w),
        "pop_zipf": zipf,
        "polysemy": poly,
        "brys": brys,
        "brys_label": brys_label,
        "pos": pos_tag,
    }

def _valid_word(word: str) -> bool:
    try:
        m = word_metrics(word)
    except Exception:
        return False
    required_fields = ["pos", "pop_zipf", "pop_rank"]
    return all(m.get(f) is not None for f in required_fields)

# -------------------------------
# Prompt builders
# -------------------------------
def _prompt(A, B):
    return PROMPT_VARIATIONS["colon"](A, B)

def _prompt_english(A, B):
    return PROMPT_VARIATIONS["english"](A, B)

def _prompt_one_shot(A, B):
    return PROMPT_VARIATIONS["one_shot"](A, B)

def _prompt_few_shot(A, B):
    return PROMPT_VARIATIONS["few_shot"](A, B)

# -------------------------------
# Run a trial
# -------------------------------
def run_trial(model: str, *args, prompt_type: str = "colon", verbose: bool = True):
    """
    If positional arguments are provided, treat them as A,B.
    Otherwise, sample automatically.

    For compatibility with the current pipeline, correctness is stored under
    the key 'is_identity', even though this is not an identity task.
    """
    rng = random.Random()
    modes = {}

    # -------------------------------
    # 1️⃣ Positional args override
    # -------------------------------
    if len(args) >= 2:
        A, B = args[:2]
        for k, v in zip("AB", (A, B)):
            modes[k] = "from_dataset"
    else:
        words = [w for w in ALL if _valid_word(w)]
        A, B = [rng.choice(words) for _ in range(2)]
        for k, v in zip("AB", (A, B)):
            modes[k] = "random"

    # -------------------------------
    # Prompt
    # -------------------------------
    prompt_fn = PROMPT_VARIATIONS.get(prompt_type, _prompt)
    prompt = prompt_fn(A, B) if callable(prompt_fn) else _prompt(A, B)

    # -------------------------------
    # Inference
    # -------------------------------
    raw = generate_inference(prompt, model)
    pred = clean_answer(raw)
    hit = (pred == B.lower())

    # -------------------------------
    # Lexical metrics
    # -------------------------------
    metrics = {
        f"{k}_metrics": word_metrics(v)
        for k, v in zip("AB", (A, B))
    }

    # -------------------------------
    # Output
    # -------------------------------
    out = {
        "model": model,
        "analogy_type": TASK_NAME,
        "A": A,
        "B": B,
        **metrics,
        **{f"{k}_mode": m for k, m in modes.items()},
        "prompt": prompt,
        "raw_response": raw,
        "parsed_answer": pred,
        "expected": B,
        "is_identity": hit,   # keep for pipeline compatibility
        "prompt_type": prompt_type,
    }

    if verbose:
        print(
            f"[unbracket] "
            f"A={A} B={B} pred='{pred}' expected='{B.lower()}' "
            f"hit={hit} prompt={prompt_type}"
        )

    return out

# -------------------------------
# Dataset generator
# -------------------------------
def generate_dataset(n: int, rng: random.Random):
    """
    Generate n fixed 2-tuples (A,B) from ALL valid words.

    Task rule:
      (A) : A :: (B) : B
    """
    words = [w for w in ALL if _valid_word(w)]

    dataset = []
    rejected = 0
    attempts = 0

    while len(dataset) < n:
        attempts += 1
        A, B = [rng.choice(words) for _ in range(2)]

        # Avoid trivial support/target identity
        if A == B:
            rejected += 1
            continue

        dataset.append((A, B))

    sampling_stats = {
        "source_vocab": "ALL",
        "initial_pool_size": len(words),
        "total_attempts": attempts,
        "total_rejected": rejected,
        "acceptance_rate": len(dataset) / attempts if attempts else None,
        "rule": "remove parentheses; expected answer is 'B'",
    }
    return dataset, sampling_stats