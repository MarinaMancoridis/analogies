# analogies/analogy_types/pair_analogy.py
import random
import re
from typing import Dict
from analogies.utils import generate_inference
from analogies.common import sample_concept, clean_answer

RNG = random.Random()

# ---- Relation catalog (concise, implementation-oriented) ----
# Each item: (key, human_name, relation_instruction)
RELATIONS = [
    ("class_inclusion", "Class-Inclusion",
     "B is a more specific instance of A (class → member/individual). e.g., flower:tulip."),

    ("part_whole", "Part-Whole",
     "B is a constituent of A (component/part) or a typical member of collection A. e.g., car:engine, forest:tree."),

    ("similar", "Similar",
     "B is similar or nearly equivalent to A (synonym or close along a dimension). e.g., car:auto; simmer:boil."),

    ("contrast", "Contrast",
     "B contrasts with A (opposite or reverse). e.g., old:young; buy:sell."),

    ("attribute", "Attribute",
     "B is a typical attribute or state of A. e.g., beggar:poor; coward:fear."),

    ("non_attribute", "Non-Attribute",
     "B is an attribute or state A typically lacks or is incompatible with. e.g., fire:cold; corpse:life."),

    ("case_relations", "Case Relations",
     "Agent/Instrument or Action/Object: for agent A, B is a typical instrument; for action A, B is a typical object/patient. e.g., soldier:gun; plow:earth."),

    ("cause_purpose", "Cause-Purpose",
     "B is the typical effect of A or an action taken in response to A. e.g., joke:laughter; hunger:eat."),

    ("space_time", "Space-Time",
     "B is typically associated with location or time A. e.g., library:book; winter:snow."),

    ("reference", "Reference",
     "A is a sign or representation referring to B. e.g., siren:danger; diary:person."),
]


# ---- Prompt builders ----
def _prompt_generate_B(A: str, relation_text: str) -> str:
    return (
        "You are given a relation description. Produce ONE English word B such that (A, B) fits the relation.\n"
        "Output ONLY: ANSWER: <word>\n\n"
        f"RELATION: {relation_text}\nA: {A}\n"
    )

def _prompt_complete_analogy(A: str, B: str, C: str, relation_text: str) -> str:
    return (
        "Complete the analogy consistent with the relation described. "
        "Output ONLY: ANSWER: <word>\n\n"
        f"RELATION: {relation_text}\n"
        f"{A} : {B} :: {C} : ____\n"
    )

def _prompt_grade(A: str, B: str, C: str, D: str, relation_text: str) -> str:
    return (
        "Grade whether BOTH pairs (A,B) and (C,D) satisfy the relation. "
        "Reply ONLY:\nGRADE: Correct or Incorrect\n"
        "REASON: <short reason>\n\n"
        f"RELATION: {relation_text}\nA:{A}  B:{B}  C:{C}  D:{D}\n"
    )

# ---- Public entrypoint (same contract as identity.run_trial) ----
def run_trial(model: str, *, verbose: bool = True) -> Dict:
    """
    Build a pair analogy by:
      1) sampling A (concept),
      2) sampling a relation, asking the LLM for B to match (A,B),
      3) sampling C (concept),
      4) asking LLM to complete D in A:B :: C:D,
      5) asking LLM to grade whether both pairs satisfy the relation.

    Returns a dict with keys compatible with the runner + infer_hit (uses 'grade_correct').
    """
    rel_key, rel_name, rel_text = RNG.choice(RELATIONS)
    if verbose:
        print(f"\n[relation] {rel_key} — {rel_name}")

    # Step 1: A
    A = sample_concept(model, verbose=verbose)
    if verbose:
        print(f"[A] {A}")

    # Step 2: B from relation(A, ?)
    pB = _prompt_generate_B(A, rel_text)
    rawB = generate_inference(pB, model)
    B = clean_answer(rawB)
    if verbose:
        print(f"[B] prompt=>\n{pB}\n[B] raw={rawB.strip()}\n[B] parsed={B}")

    # Step 3: C (ensure variety)
    C = sample_concept(model, verbose=verbose)
    while C == A:
        C = sample_concept(model, verbose=verbose)
    if verbose:
        print(f"[C] {C}")

    # Step 4: D from analogy completion under the relation
    pD = _prompt_complete_analogy(A, B, C, rel_text)
    rawD = generate_inference(pD, model)
    D = clean_answer(rawD)
    if verbose:
        print(f"[D] prompt=>\n{pD}\n[D] raw={rawD.strip()}\n[D] parsed={D}")

    # Step 5: grade both pairs under the relation
    pG = _prompt_grade(A, B, C, D, rel_text)
    rawG = generate_inference(pG, model)
    is_correct = bool(re.search(r"\bGRADE:\s*Correct\b", rawG, re.IGNORECASE))
    if verbose:
        print(f"[grade] prompt=>\n{pG}\n[grade] raw={rawG.strip()}\n[grade] correct={is_correct}")

    return {
        "model": model,
        "analogy_type": "pair",
        "relation_key": rel_key,
        "relation_name": rel_name,
        "relation_text": rel_text,
        "A": A, "B": B, "C": C, "D": D,
        "prompts": {"B": pB, "D": pD, "grade": pG},
        "raw": {"B": rawB, "D": rawD, "grade": rawG},
        "grade_correct": is_correct,  # <-- infer_hit() will pick this up
    }
