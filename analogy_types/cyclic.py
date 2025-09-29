from analogies.utils import generate_inference
from analogies.common import sample_concept, clean_answer

def _prompt(A: str, B: str, C: str, D: str) -> str:
    return (
        "Follow the same transformation on both sides. "
        "Reply ONLY as: ANSWER: <word>\n\n"
        f"{A} → {B} → {A} :: {C} → {D} → ____"
    )

def run_trial(model: str, *, verbose: bool = True):
    A = sample_concept(model, verbose=verbose)
    B = sample_concept(model, verbose=verbose)
    while B == A:
        B = sample_concept(model, verbose=verbose)

    C = sample_concept(model, verbose=verbose)
    D = sample_concept(model, verbose=verbose)
    while D == C:
        D = sample_concept(model, verbose=verbose)

    prompt = _prompt(A, B, C, D)
    if verbose:
        print(f"[cyclic] A={A} B={B} C={C} D={D}\nPrompt:\n{prompt}")

    raw = generate_inference(prompt, model)
    pred = clean_answer(raw)
    hit = (pred == C.lower())  # identity if it returns to C

    if verbose:
        print(f"[cyclic] raw={raw.strip()}\nparsed={pred} expected={C} hit={hit}")

    return {
        "model": model,
        "analogy_type": "cyclic",
        "A": A, "B": B, "C": C, "D": D,
        "prompt": prompt,
        "raw_response": raw,
        "parsed_answer": pred,
        "expected": C,
        "is_identity": hit
    }
