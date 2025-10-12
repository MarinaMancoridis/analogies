from analogies.utils import generate_inference
from analogies.common import sample_concept, clean_answer, _get_picker
from typing import List, Dict, Tuple, Optional
import math

def _prompt(A: str, C: str) -> str:
    return f"Complete the analogy. Reply ONLY as: ANSWER: <word>\n\n{A} : {A} :: {C} : ____"

def run_trial(model: str, *, A_mode: str, C_mode: str, verbose: bool = True):
    A_picker = _get_picker(A_mode)
    C_picker = _get_picker(C_mode)

    # Only what we need: A and C; each retried until it is a real concept.
    A = sample_concept(model, picker=A_picker, verbose=verbose)
    C = sample_concept(model, picker=C_picker, verbose=verbose)  # C==A allowed

    prompt = _prompt(A, C)
    if verbose:
        print(f"[identity] A={A} C={C}\nPrompt:\n{prompt}")

    raw = generate_inference(prompt, model)
    pred = clean_answer(raw)
    hit = (pred == C.lower())

    if verbose:
        print(f"[identity] raw={raw.strip()}\nparsed={pred} expected={C} hit={hit}")

    return {
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

