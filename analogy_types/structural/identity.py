from analogies.utils import generate_inference
from analogies.common import sample_concept, clean_answer, _get_picker, ALL, _brys_score, ac_label_brys
from typing import List, Dict, Optional
import random
import nltk
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency
from analogies.analogy_types.pair_analogy import RELATIONS, _pick_from_dict_for_role, _prompt_generate_B, _ask_one, _fallback_C, RNG

# -------------------------------
# Task name
# -------------------------------
TASK_NAME = "identity"

# -------------------------------
# Prompt variations
# -------------------------------
PROMPT_VARIATIONS = {
    "colon": lambda A, C: f"Complete the analogy. Reply ONLY as: ANSWER: <word>\n\n{A} : {A} :: {C} : ____",
    "english": lambda A, C: f"Complete the analogy. Reply ONLY as: ANSWER: <word>\n\n{A} is to {A} as {C} is to ____",
    "one_shot": lambda A, C: f"Complete the analogy. Reply ONLY as: ANSWER: <word>\n\n{A} : {A} :: {C} : ____. \n\nHere is one example. \nrose : rose :: flower : flower",
    "few_shot": lambda A, C: f"Complete the analogy. Reply ONLY as: ANSWER: <word>\n\n{A} : {A} :: {C} : ____. \n\nHere are some examples. \nrose : rose :: flower : flower \napple : apple :: fruit : fruit \ncat : cat :: dog : dog",
}

# -------------------------------
# Word metrics
# -------------------------------
_RANK_INDEX: Optional[Dict[str, int]] = None

def _ensure_rank_index() -> None:
    global _RANK_INDEX
    if _RANK_INDEX is not None:
        return
    _RANK_INDEX = {w.lower(): i+1 for i, w in enumerate(ALL)}

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
# Relationship sampling
# -------------------------------
def _ac_from_relation(model: str, rel_key: Optional[str] = None):
    rel = (next(r for r in RELATIONS if r[0] == rel_key) if rel_key
           else RNG.choice(RELATIONS))
    rk, rn, rtext, arole = rel
    A, _, _ = _pick_from_dict_for_role(model, arole, rtext, avoid=set())
    C, _ = _ask_one(model, _prompt_generate_B(A, rtext))
    if not C or C == A:
        C = _fallback_C(arole, {A})
    return A, C, {"relation_key": rk, "relation_name": rn, "relation_text": rtext}

# -------------------------------
# Prompt builders
# -------------------------------
def _prompt(A, C): return PROMPT_VARIATIONS["colon"](A, C)
def _prompt_english(A, C): return PROMPT_VARIATIONS["english"](A, C)
def _prompt_one_shot(A, C): return PROMPT_VARIATIONS["one_shot"](A, C)
def _prompt_few_shot(A, C): return PROMPT_VARIATIONS["few_shot"](A, C)

# -------------------------------
# Run a trial
# -------------------------------
def run_trial(model: str, *args, A_mode: str = "random", C_mode: str = "random",
              prompt_type: str = "colon", verbose: bool = True):
    """
    If positional arguments are provided, treat as A,C.
    Otherwise, sample automatically based on modes.
    Stores lexical metrics for each word and logs actual source of each word.
    """
    relmeta = domainmeta = None

    # Track actual mode used
    A_mode_used = C_mode_used = None

    # -------------------------------
    # 1️⃣ Positional args override
    # -------------------------------
    if len(args) >= 2:
        A, C = args[:2]
        A_mode_used = "from_dataset"
        C_mode_used = "from_dataset"

    # -------------------------------
    # 2️⃣ Relation sampling
    # -------------------------------
    elif A_mode.startswith("rel"):
        rel_key = A_mode.split(":", 1)[1] if ":" in A_mode else None
        A, C, relmeta = _ac_from_relation(model, rel_key)
        A_mode_used = A_mode
        C_mode_used = C_mode

    # -------------------------------
    # 3️⃣ Fully automatic sampling
    # -------------------------------
    else:
        A_picker = _get_picker(A_mode)
        C_picker = _get_picker(C_mode)

        if getattr(A_picker, "_joint", None):
            A, C = A_picker()
            domainmeta = getattr(A_picker, "_meta", None)
        elif getattr(C_picker, "_joint", None):
            A, C = C_picker()
        else:
            A, C = A_picker(), C_picker()

        A_mode_used = A_mode
        C_mode_used = C_mode

    # -------------------------------
    # Prompt
    # -------------------------------
    prompt_fn = PROMPT_VARIATIONS.get(prompt_type, _prompt)
    prompt = prompt_fn(A, C) if callable(prompt_fn) else _prompt(A, C)

    # -------------------------------
    # Inference
    # -------------------------------
    raw = generate_inference(prompt, model)
    pred = clean_answer(raw)
    hit = (pred == C.lower())

    # -------------------------------
    # Output with lexical metrics
    # -------------------------------
    out = {
        "model": model,
        "analogy_type": TASK_NAME,
        "A_mode": A_mode_used,
        "C_mode": C_mode_used,
        "A": A,
        "B": None,
        "C": C,
        "D": None,
        "A_metrics": word_metrics(A),
        "C_metrics": word_metrics(C),
        "prompt": prompt,
        "raw_response": raw,
        "parsed_answer": pred,
        "expected": C,
        "is_identity": hit,
        "prompt_type": prompt_type,
    }

    if relmeta: out["_relmeta"] = relmeta
    if domainmeta: out["_domainmeta"] = domainmeta

    if verbose:
        print(f"[identity] A={A} C={C} hit={hit} A_mode={A_mode_used} C_mode={C_mode_used} prompt={prompt_type}")

    return out

# -------------------------------
# Dataset generator
# -------------------------------
def generate_dataset(n: int, rng: random.Random):
    """
    Generate n (A,C) tuples from ALL words that are valid.
    """
    words = [w for w in ALL if _valid_word(w)]
    dataset = []
    rejected = 0
    attempts = 0
    while len(dataset) < n:
        attempts += 1
        A, C = rng.choice(words), rng.choice(words)
        if A == C:
            rejected += 1
            continue
        dataset.append((A, C))
    sampling_stats = {
        "source_vocab": "ALL",
        "initial_pool_size": len(words),
        "total_attempts": attempts,
        "total_rejected": rejected,
        "acceptance_rate": len(dataset)/attempts if attempts else None
    }
    return dataset, sampling_stats