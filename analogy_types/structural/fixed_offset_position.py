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
TASK_NAME = "fixed_offset_position"

# -------------------------------
# Prompt variations
# Pattern: A B C D : B C :: E F G H : ____
# Rule: keep positions 2 and 3
# Expected answer: F G
# -------------------------------
PROMPT_VARIATIONS = {
    "colon": lambda A, B, C, D, E, F, G, H: (
        f"Complete the analogy. Reply ONLY as: ANSWER: <word1> <word2>\n\n"
        f"{A} {B} {C} {D} : {B} {C} :: {E} {F} {G} {H} : ____"
    ),
    "english": lambda A, B, C, D, E, F, G, H: (
        f"Complete the pattern. Reply ONLY as: ANSWER: <word1> <word2>\n\n"
        f"The second sequence keeps the 2nd and 3rd words from the first sequence.\n"
        f"{A} {B} {C} {D} : {B} {C} :: {E} {F} {G} {H} : ____"
    ),
    "one_shot": lambda A, B, C, D, E, F, G, H: (
        f"Complete the analogy. Reply ONLY as: ANSWER: <word1> <word2>\n\n"
        f"{A} {B} {C} {D} : {B} {C} :: {E} {F} {G} {H} : ____\n\n"
        f"Here is one example:\n"
        f"red blue green yellow : blue green :: cat dog bird fish : dog bird"
    ),
    "few_shot": lambda A, B, C, D, E, F, G, H: (
        f"Complete the analogy. Reply ONLY as: ANSWER: <word1> <word2>\n\n"
        f"{A} {B} {C} {D} : {B} {C} :: {E} {F} {G} {H} : ____\n\n"
        f"Here are some examples:\n"
        f"red blue green yellow : blue green :: cat dog bird fish : dog bird\n"
        f"cup plate bowl spoon : plate bowl :: sun moon star cloud : moon star"
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
# Output parsing for 2-word answers
# -------------------------------
def _normalize_token(tok: str) -> str:
    tok = tok.lower().strip()
    tok = re.sub(r"^[^\w]+|[^\w]+$", "", tok)
    return tok

def _extract_two_word_answer(raw: str) -> str:
    """
    Try to robustly extract a two-word answer from model output.
    Returns normalized form: 'word1 word2'
    """
    if raw is None:
        return ""

    text = raw.strip()

    m = re.search(r"ANSWER\s*:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        text = m.group(1).strip()

    parts = re.split(r"[\s,|/]+", text)
    parts = [_normalize_token(p) for p in parts]
    parts = [p for p in parts if p]

    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}"
    elif len(parts) == 1:
        return parts[0]
    return ""

# -------------------------------
# Prompt builders
# -------------------------------
def _prompt(A, B, C, D, E, F, G, H):
    return PROMPT_VARIATIONS["colon"](A, B, C, D, E, F, G, H)

def _prompt_english(A, B, C, D, E, F, G, H):
    return PROMPT_VARIATIONS["english"](A, B, C, D, E, F, G, H)

def _prompt_one_shot(A, B, C, D, E, F, G, H):
    return PROMPT_VARIATIONS["one_shot"](A, B, C, D, E, F, G, H)

def _prompt_few_shot(A, B, C, D, E, F, G, H):
    return PROMPT_VARIATIONS["few_shot"](A, B, C, D, E, F, G, H)

# -------------------------------
# Run a trial
# -------------------------------
def run_trial(model: str, *args, prompt_type: str = "colon", verbose: bool = True):
    """
    If positional arguments are provided, treat them as A,B,C,D,E,F,G,H.
    Otherwise, sample automatically.

    For compatibility with the current pipeline, correctness is stored under
    the key 'is_identity', even though this is not an identity task.
    """
    rng = random.Random()
    modes = {}

    # -------------------------------
    # 1️⃣ Positional args override
    # -------------------------------
    if len(args) >= 8:
        A, B, C, D, E, F, G, H = args[:8]
        for k, v in zip("ABCDEFGH", (A, B, C, D, E, F, G, H)):
            modes[k] = "from_dataset"
    else:
        words = [w for w in ALL if _valid_word(w)]
        A, B, C, D, E, F, G, H = [rng.choice(words) for _ in range(8)]
        for k, v in zip("ABCDEFGH", (A, B, C, D, E, F, G, H)):
            modes[k] = "random"

    # -------------------------------
    # Prompt
    # -------------------------------
    prompt_fn = PROMPT_VARIATIONS.get(prompt_type, _prompt)
    prompt = prompt_fn(A, B, C, D, E, F, G, H) if callable(prompt_fn) else _prompt(A, B, C, D, E, F, G, H)

    # -------------------------------
    # Inference
    # -------------------------------
    raw = generate_inference(prompt, model)

    parsed_single = clean_answer(raw)
    parsed_pair = _extract_two_word_answer(raw)

    expected = f"{F.lower()} {G.lower()}"
    hit = (parsed_pair == expected)

    # -------------------------------
    # Lexical metrics
    # -------------------------------
    metrics = {
        f"{k}_metrics": word_metrics(v)
        for k, v in zip("ABCDEFGH", (A, B, C, D, E, F, G, H))
    }

    # -------------------------------
    # Output
    # -------------------------------
    out = {
        "model": model,
        "analogy_type": TASK_NAME,
        "A": A, "B": B, "C": C, "D": D,
        "E": E, "F": F, "G": G, "H": H,
        **metrics,
        **{f"{k}_mode": m for k, m in modes.items()},
        "prompt": prompt,
        "raw_response": raw,
        "parsed_answer": parsed_pair,
        "parsed_answer_single": parsed_single,
        "expected": expected,
        "is_identity": hit,   # keep for pipeline compatibility
        "prompt_type": prompt_type,
    }

    if verbose:
        print(
            f"[fixed_offset_position] "
            f"A={A} B={B} C={C} D={D} E={E} F={F} G={G} H={H} "
            f"pred='{parsed_pair}' expected='{expected}' "
            f"hit={hit} prompt={prompt_type}"
        )

    return out

# -------------------------------
# Dataset generator
# -------------------------------
def generate_dataset(n: int, rng: random.Random):
    """
    Generate n fixed 8-tuples (A,B,C,D,E,F,G,H) from ALL valid words.

    Task rule:
      A B C D : B C :: E F G H : F G
    """
    words = [w for w in ALL if _valid_word(w)]

    dataset = []
    rejected = 0
    attempts = 0

    while len(dataset) < n:
        attempts += 1
        A, B, C, D, E, F, G, H = [rng.choice(words) for _ in range(8)]

        # Avoid degenerate sequences where positions 2 and 3 collapse
        if len({A, B, C, D}) < 4 or len({E, F, G, H}) < 4:
            rejected += 1
            continue

        dataset.append((A, B, C, D, E, F, G, H))

    sampling_stats = {
        "source_vocab": "ALL",
        "initial_pool_size": len(words),
        "total_attempts": attempts,
        "total_rejected": rejected,
        "acceptance_rate": len(dataset) / attempts if attempts else None,
        "rule": "keep positions 2 and 3; expected answer is 'F G'",
    }
    return dataset, sampling_stats