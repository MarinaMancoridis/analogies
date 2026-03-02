from analogies.utils import generate_inference
from analogies.common import sample_concept, clean_answer, _get_picker
from typing import List, Dict, Tuple, Optional
import math
from analogies.analogy_types.pair_analogy import (
    RELATIONS, _pick_from_dict_for_role, _prompt_generate_B, _ask_one, _fallback_C, RNG
)

# -------------------------------
# Prompt variations for robustness
# -------------------------------
PROMPT_VARIATIONS = {
    "colon": lambda A, C: f"Complete the analogy. Reply ONLY as: ANSWER: <word>\n\n{A} : {A} :: {C} : ____",
    "english": lambda A, C: f"Complete the analogy. Reply ONLY as: ANSWER: <word>\n\n{A} is to {A} as {C} is to ____",
    "one_shot": lambda A, C: f"Complete the analogy. Reply ONLY as: ANSWER: <word>\n\n{A} : {A} :: {C} : ____. \n\nHere is one example. \nrose : rose :: flower : flower",
    "few_shot": lambda A, C: f"Complete the analogy. Reply ONLY as: ANSWER: <word>\n\n{A} : {A} :: {C} : ____. \n\nHere are some examples. \nrose : rose :: flower : flower \napple : apple :: fruit : fruit \ncat : cat :: dog : dog",
}

# ----- used for relationship type sampling -----
def _ac_from_relation(model: str, rel_key: str | None = None):
    rel = (next(r for r in RELATIONS if r[0] == rel_key) if rel_key
           else RNG.choice(RELATIONS))  # ← random relation if None
    rk, rn, rtext, arole = rel
    # A via your role validator on dictionary words
    A, _, _ = _pick_from_dict_for_role(model, arole, rtext, avoid=set())
    # C via your relation-aware “generate B” prompt
    C, _ = _ask_one(model, _prompt_generate_B(A, rtext))
    if (not C) or (C == A):
        C = _fallback_C(arole, {A})
    return A, C, {"relation_key": rk, "relation_name": rn, "relation_text": rtext}



# ------------- Prompt builders -------------
def _prompt(A: str, C: str) -> str:
    return f"Complete the analogy. Reply ONLY as: ANSWER: <word>\n\n{A} : {A} :: {C} : ____"

def _prompt_english(A: str, C: str) -> str:
    return f"Complete the analogy. Reply ONLY as: ANSWER: <word>\n\n{A} is to {A} as {C} is to ____"

def _prompt_one_shot(A: str, C: str) -> str:
    return f"Complete the analogy. Reply ONLY as: ANSWER: <word>\n\n{A} : {A} :: {C} : ____. \n\nHere is one example. \nrose : rose :: flower : flower"

def _prompt_few_shot(A: str, C: str) -> str:
    return f"Complete the analogy. Reply ONLY as: ANSWER: <word>\n\n{A} : {A} :: {C} : ____. \n\nHere are some examples. \nrose : rose :: flower : flower \napple : apple :: fruit : fruit \ncat : cat :: dog : dog"

# ------------- UPDATED run_trial -------------
def run_trial(
    model: str, *,
    A_mode: str,
    C_mode: str,
    prompt_type: str = "colon",  # default prompt
    verbose: bool = True
):
    """
    Run a single trial for a given model and (A,C) sampling mode.
    Supports multiple prompt types for robustness testing.
    """
    relmeta = None
    domainmeta = None

    # Sample A,C pair
    if A_mode.startswith("rel"):
        rel_key = A_mode.split(":", 1)[1] if ":" in A_mode else None
        A, C, relmeta = _ac_from_relation(model, rel_key)
    else:
        A_picker = _get_picker(A_mode)
        if getattr(A_picker, "_joint", None):
            A, C = A_picker()
            if getattr(A_picker, "_joint", "") in ("domain_same", "domain_diff"):
                domainmeta = getattr(A_picker, "_meta", None)
        else:
            C_picker = _get_picker(C_mode)
            if getattr(C_picker, "_joint", None):
                A, C = C_picker()
            elif A_mode == "random" and C_mode == "random":
                A = A_picker()
                C = C_picker()
            else:
                A = A_picker()
                C = C_picker()
            relmeta = None

    # Build prompt according to the requested prompt_type
    prompt_fn = PROMPT_VARIATIONS.get(prompt_type, _prompt)
    prompt = prompt_fn(A, C) if callable(prompt_fn) else _prompt(A, C)

    # Generate and clean model response
    raw = generate_inference(prompt, model)
    pred = clean_answer(raw)
    hit = (pred == C.lower())

    out = {
        "model": model,
        "analogy_type": "identity",
        "A_mode": A_mode,
        "C_mode": C_mode,
        "A": A,
        "B": None,
        "C": C,
        "D": None,
        "prompt": prompt,
        "raw_response": raw,
        "parsed_answer": pred,
        "expected": C,
        "is_identity": hit,
        "prompt_type": prompt_type
    }

    if relmeta:
        out["_relmeta"] = relmeta
    if domainmeta:
        out["_domainmeta"] = domainmeta

    if verbose:
        print(f"[identity] A={A} C={C} hit={hit} rel={out.get('relation_key')} prompt={prompt_type}")

    return out