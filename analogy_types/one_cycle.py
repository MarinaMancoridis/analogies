from analogies.utils import generate_inference
from analogies.common import sample_concept, clean_answer, _get_picker, ALL, _brys_score, ac_label_brys
from typing import List, Dict, Optional
from analogies.analogy_types.pair_analogy import RELATIONS, _pick_from_dict_for_role, _prompt_generate_B, _ask_one, _fallback_C, RNG
import random
import nltk
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency

# -------------------------------
# Task name
# -------------------------------
TASK_NAME = "one_cycle"

# -------------------------------
# Prompt variations
# -------------------------------
PROMPT_VARIATIONS = {
    "colon": lambda A,B,C,D,E,F,G,H: (
        f"Complete the 1-cycle analogy. Reply ONLY as: ANSWER: <word>\n\n"
        f"{A} -> {B} -> {A} : {C} -> {D} -> {C} :: {E} -> {F} -> {E} : {G} -> {H} -> ____"
    ),
    "english": lambda A,B,C,D,E,F,G,H: (
        f"Complete the 1-cycle analogy. Reply ONLY as: ANSWER: <word>\n\n"
        f"{A} goes to {B} and back to {A}, {C} goes to {D} and back to {C}, "
        f"{E} goes to {F} and back to {E}, {G} goes to {H} and back to ____"
    ),
    "one_shot": lambda A,B,C,D,E,F,G,H: (
        f"Complete the 1-cycle analogy. Reply ONLY as: ANSWER: <word>\n\n"
        f"{A} -> {B} -> {A} : {C} -> {D} -> {C} :: {E} -> {F} -> {E} : {G} -> {H} -> ____\n\n"
        f"Here is one example:\nrose -> flower -> rose : apple -> fruit -> apple :: cat -> pet -> cat : dog -> animal -> ____"
    ),
    "few_shot": lambda A,B,C,D,E,F,G,H: (
        f"Complete the 1-cycle analogy. Reply ONLY as: ANSWER: <word>\n\n"
        f"{A} -> {B} -> {A} : {C} -> {D} -> {C} :: {E} -> {F} -> {E} : {G} -> {H} -> ____\n\n"
        f"Here are some examples:\n"
        f"rose -> flower -> rose : apple -> fruit -> apple :: cat -> pet -> cat : dog -> animal -> dog\n"
        f"sun -> light -> sun : moon -> night -> moon :: bird -> wing -> bird : fish -> fin -> ____"
    ),
}

# -------------------------------
# Word metrics
# -------------------------------
_RANK_INDEX: Optional[Dict[str,int]] = None

def _ensure_rank_index() -> None:
    global _RANK_INDEX
    if _RANK_INDEX is not None:
        return
    _RANK_INDEX = {w.lower(): i+1 for i,w in enumerate(ALL)}

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

# -------------------------------
# Relationship sampling
# -------------------------------
def _abcd_from_relation(model: str, rel_key: Optional[str]=None):
    rel = next((r for r in RELATIONS if r[0]==rel_key), None) or RNG.choice(RELATIONS)
    rk, rn, rtext, arole = rel
    A, _, _ = _pick_from_dict_for_role(model, arole, rtext, avoid=set())
    B, _ = _ask_one(model, _prompt_generate_B(A, rtext))
    if not B: B = _fallback_C(arole, {A})
    C, _, _ = _pick_from_dict_for_role(model, arole, rtext, avoid={A,B})
    D, _ = _ask_one(model, _prompt_generate_B(C, rtext))
    if not D: D = _fallback_C(arole, {C})
    return (A,B,C,D), {"relation_key": rk, "relation_name": rn, "relation_text": rtext}

# -------------------------------
# Prompt builders
# -------------------------------
def _prompt(A,B,C,D,E,F,G,H): return PROMPT_VARIATIONS["colon"](A,B,C,D,E,F,G,H)
def _prompt_english(A,B,C,D,E,F,G,H): return PROMPT_VARIATIONS["english"](A,B,C,D,E,F,G,H)
def _prompt_one_shot(A,B,C,D,E,F,G,H): return PROMPT_VARIATIONS["one_shot"](A,B,C,D,E,F,G,H)
def _prompt_few_shot(A,B,C,D,E,F,G,H): return PROMPT_VARIATIONS["few_shot"](A,B,C,D,E,F,G,H)

# -------------------------------
# Run a trial
# -------------------------------
def run_trial(model: str, *args, prompt_type: str="colon", verbose: bool=True):
    relmeta = None
    rng = random.Random()

    # Track modes
    modes = {}

    # -------------------------------
    # 1️⃣ Positional args override
    # -------------------------------
    if len(args) >= 8:
        A,B,C,D,E,F,G,H = args[:8]
        for k,v in zip("ABCDEFGH",(A,B,C,D,E,F,G,H)):
            modes[k] = "from_dataset"
    else:
        # Sample A,B,C,D from relation
        (A,B,C,D), relmeta = _abcd_from_relation(model)
        for k,v in zip("ABCD",(A,B,C,D)):
            modes[k] = "relation"
        # Sample E,F,G,H randomly
        E, F, G, H = rng.choice(ALL), rng.choice(ALL), rng.choice(ALL), rng.choice(ALL)
        for k,v in zip("EFGH",(E,F,G,H)):
            modes[k] = "random"

    # Prompt
    prompt_fn = PROMPT_VARIATIONS.get(prompt_type, _prompt)
    prompt = prompt_fn(A,B,C,D,E,F,G,H) if callable(prompt_fn) else _prompt(A,B,C,D,E,F,G,H)

    # Inference
    raw = generate_inference(prompt, model)
    pred = clean_answer(raw)
    hit = pred == G.lower()

    # Lexical metrics
    metrics = {f"{k}_metrics": word_metrics(v) for k,v in zip("ABCDEFGH",(A,B,C,D,E,F,G,H))}

    # Output
    out = {
        "model": model,
        "analogy_type": TASK_NAME,
        "A": A, "B": B, "C": C, "D": D,
        "E": E, "F": F, "G": G, "H": H,
        **metrics,
        **{f"{k}_mode": m for k,m in modes.items()},
        "prompt": prompt,
        "raw_response": raw,
        "parsed_answer": pred,
        "expected": G,
        "is_identity": hit,
        "prompt_type": prompt_type,
    }

    if relmeta: out["_relmeta"] = relmeta
    if verbose:
        print(f"[one_cycle] A={A} B={B} C={C} D={D} E={E} F={F} G={G} H={H} prompt={prompt_type}")

    return out

# -------------------------------
# Dataset generator
# -------------------------------
def generate_dataset(n: int, rng: random.Random):
    words = [w for w in ALL if isinstance(w,str)]
    dataset = []
    rejected = 0
    attempts = 0
    while len(dataset) < n:
        attempts += 1
        # A,B,C,D from relation sampling
        A,B,C,D = RNG.choice(words), RNG.choice(words), RNG.choice(words), RNG.choice(words)
        # E,F,G,H randomly
        E,F,G,H = rng.choice(words), rng.choice(words), rng.choice(words), rng.choice(words)
        dataset.append((A,B,C,D,E,F,G,H))
    sampling_stats = {
        "source_vocab": "ALL",
        "initial_pool_size": len(words),
        "total_attempts": attempts,
        "total_rejected": rejected,
        "acceptance_rate": len(dataset)/attempts if attempts else None
    }
    return dataset, sampling_stats