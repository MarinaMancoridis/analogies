from analogies.utils import generate_inference
from analogies.common import sample_concept, clean_answer

def _prompt(A: str, C: str) -> str:
    return f"Complete the analogy. Reply ONLY as: ANSWER: <word>\n\n{A} : {A} :: {C} : ____"

def run_trial(model: str, *, verbose: bool = True):
    # Only what we need: A and C; each retried until it is a real concept.
    A = sample_concept(model, verbose=verbose)
    C = sample_concept(model, verbose=verbose)  # allow C==A; still a valid identity test

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
        "A": A, "B": None, "C": C, "D": None,
        "prompt": prompt,
        "raw_response": raw,
        "parsed_answer": pred,
        "expected": C,
        "is_identity": hit
    }
