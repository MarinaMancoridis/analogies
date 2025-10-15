from analogies.utils import generate_inference
from analogies.common import sample_concept, clean_answer, _get_picker
from typing import List, Dict, Tuple, Optional
import math
from analogies.analogy_types.pair_analogy import (
    RELATIONS, _pick_from_dict_for_role, _prompt_generate_B, _ask_one, _fallback_C, RNG
)

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

# ------------- UPDATED run_trial -------------
def run_trial(model: str, *, A_mode: str, C_mode: str, verbose: bool = True):
    relmeta = None  # keep defined

    if A_mode.startswith("rel"):
        rel_key = A_mode.split(":", 1)[1] if ":" in A_mode else None
        A, C, relmeta = _ac_from_relation(model, rel_key)
    else:
        A_picker = _get_picker(A_mode)
        if getattr(A_picker, "_joint", None):       # ← check A first
            A, C = A_picker()                       # (A, C) from joint picker
        else:
            C_picker = _get_picker(C_mode)          # only look up C if needed
            if getattr(C_picker, "_joint", None):
                A, C = C_picker()
            else:
                A = sample_concept(model, picker=A_picker, verbose=verbose)
                C = sample_concept(model, picker=C_picker, verbose=verbose)
            relmeta = None

    prompt = _prompt(A, C)
    raw = generate_inference(prompt, model)
    pred = clean_answer(raw)
    hit = (pred == C.lower())

    out = {
        "model": model,
        "analogy_type": "identity",
        "A_mode": A_mode, "C_mode": C_mode,
        "A": A, "B": None, "C": C, "D": None,
        "prompt": prompt,
        "raw_response": raw,
        "parsed_answer": pred,
        "expected": C,
        "is_identity": hit
    }
    if relmeta:
        out["_relmeta"] = relmeta
    if verbose:
        print(f"[identity] A={A} C={C} hit={hit} rel={out.get('relation_key')}")
    return out
