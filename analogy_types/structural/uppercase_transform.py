from analogies.utils import generate_inference
from analogies.common import clean_answer, ALL, _brys_score, ac_label_brys
from typing import Dict, Optional
import random
import re
import nltk
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency

# -------------------------------
# Task name
# -------------------------------
TASK_NAME = "uppercase_transform"

# -------------------------------
# Prompt variations
# Pattern: dog : DOG :: cat : ____
# Rule: convert to uppercase
# Expected answer: CAT
# -------------------------------
PROMPT_VARIATIONS = {
    "colon": lambda A, B: (
        f"Complete the analogy. Reply ONLY as: ANSWER: <word>\n\n"
        f"{A} : {A.upper()} :: {B} : ____"
    ),
    "english": lambda A, B: (
        f"Complete the pattern. Reply ONLY as: ANSWER: <word>\n\n"
        f"The second word is the uppercase version of the first.\n"
        f"{A} : {A.upper()} :: {B} : ____"
    ),
    "one_shot": lambda A, B: (
        f"Complete the analogy. Reply ONLY as: ANSWER: <word>\n\n"
        f"{A} : {A.upper()} :: {B} : ____\n\n"
        f"Here is one example:\n"
        f"dog : DOG :: cat : CAT"
    ),
    "few_shot": lambda A, B: (
        f"Complete the analogy. Reply ONLY as: ANSWER: <word>\n\n"
        f"{A} : {A.upper()} :: {B} : ____\n\n"
        f"Here are some examples:\n"
        f"dog : DOG :: cat : CAT\n"
        f"tree : TREE :: bird : BIRD"
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
# Robust answer extraction
# -------------------------------
def _extract_word(raw: str) -> str:
    if raw is None:
        return ""

    text = raw.strip()

    m = re.search(r"ANSWER\s*:\s*(.*)", text, flags=re.IGNORECASE)
    if m:
        text = m.group(1).strip()

    token = re.split(r"[\s,|/]+", text)[0]
    token = re.sub(r"^[^\w]+|[^\w]+$", "", token)

    return token

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

    parsed = _extract_word(raw)
    expected = B.upper()

    hit = (parsed == expected)

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
        "parsed_answer": parsed,
        "expected": expected,
        "is_identity": hit,  # pipeline compatibility
        "prompt_type": prompt_type,
    }

    if verbose:
        print(
            f"[uppercase_transform] "
            f"A={A} B={B} pred='{parsed}' expected='{expected}' "
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
      A : A.upper() :: B : B.upper()
    """

    words = [w for w in ALL if _valid_word(w)]

    dataset = []
    rejected = 0
    attempts = 0

    while len(dataset) < n:
        attempts += 1
        A, B = [rng.choice(words) for _ in range(2)]

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
        "rule": "convert word to uppercase; expected answer is 'B.upper()'",
    }

    return dataset, sampling_stats