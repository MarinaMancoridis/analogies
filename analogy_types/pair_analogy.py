# analogies/analogy_types/pair_analogy.py
import random
import re
from typing import Dict, Tuple
from analogies.utils import generate_inference
from analogies.common import clean_answer, WORDS

RNG = random.Random()
_SINGLE_TOKEN = re.compile(r"^[a-z]+$")

# ---- Relation catalog ----
# Each item: (key, human_name, relation_instruction, a_role_hint)
RELATIONS = [
    ("class_inclusion", "Class-Inclusion",
     "B is a more specific instance of A (class → member/individual). e.g., flower:tulip.",
     "a broad class or category (not a specific instance, not a verb)"),

    ("part_whole", "Part-Whole",
     "B is a constituent of A (component/part) or a typical member of collection A. e.g., car:engine; forest:tree.",
     "a whole object or a collection (not the part)"),

    ("similar", "Similar",
     "B is similar or nearly equivalent to A (synonym or close along a dimension). e.g., car:auto; simmer:boil.",
     "a common word that has a close synonym"),

    ("contrast", "Contrast",
     "B contrasts with A (opposite or reverse). e.g., old:young; buy:sell.",
     "a common word that has a clear opposite"),

    ("attribute", "Attribute",
     "B is a typical attribute or state of A. e.g., beggar:poor; coward:fear.",
     "an entity (person/thing) that naturally bears attributes"),

    ("non_attribute", "Non-Attribute",
     "B is an attribute or state A typically lacks or is incompatible with. e.g., fire:cold; corpse:life.",
     "an entity (thing/process) that clearly lacks some attributes"),

    ("case_relations", "Case Relations",
     "Agent/Instrument or Action/Object: e.g., soldier:gun; plow:earth.",
     "either an agent (doer) or an action verb"),

    ("cause_purpose", "Cause-Purpose",
     "B is the typical effect of A or an action taken in response to A. e.g., joke:laughter; hunger:eat.",
     "a cause/process/state that has a typical effect/purpose"),

    ("space_time", "Space-Time",
     "B is typically associated with location or time A. e.g., library:book; winter:snow.",
     "a location or a time period (not the associated thing)"),

    ("reference", "Reference",
     "A is a sign or representation referring to B. e.g., siren:danger; diary:person.",
     "a sign, record, or symbol that refers to something"),
]

# ---- Common rules text ----
SIMPLE_WORD_RULES = (
    "Rules:\n"
    "- Output exactly one lowercase English word.\n"
    "- No punctuation, no spaces, no hyphens, no numbers, no proper nouns.\n"
    "Output ONLY: ANSWER: <word>\n"
)

def _prompt_validate_role(word: str, a_role_hint: str, relation_text: str) -> str:
    return (
        "Does the WORD fit the ROLE such that it can serve as A for the RELATION?\n"
        "Reply ONLY:\n"
        "VALID: Yes or No\n"
        "REASON: <short>\n\n"
        f"WORD: {word}\nROLE: {a_role_hint}\nRELATION: {relation_text}\n"
    )

def _pick_from_dict_for_role(model: str, a_role_hint: str, relation_text: str, avoid: set[str], max_tries: int = 500):
    # Resample from dictionary until validator says VALID: Yes (or fallback)
    for _ in range(max_tries):
        cand = RNG.choice(WORDS)
        if cand in avoid or not _SINGLE_TOKEN.match(cand):
            continue
        p = _prompt_validate_role(cand, a_role_hint, relation_text)
        raw = generate_inference(p, model)
        if re.search(r"\bVALID:\s*Yes\b", raw, re.IGNORECASE):
            return cand, p, raw
    # fallback if validator never agrees
    fallback = _fallback_C(a_role_hint, avoid)
    p = _prompt_validate_role(fallback, a_role_hint, relation_text)
    raw = generate_inference(p, model)
    return fallback, p, raw

# ---- Prompt builders ----
def _prompt_generate_A(rel_key: str, relation_text: str, a_role_hint: str) -> str:
    extra = ""
    if rel_key == "class_inclusion":
        extra = "- Do NOT output verbs.\n- Prefer general categories (e.g., animal, vehicle).\n"
    return (
        "Pick a word A that fits the requested ROLE so the relation below is meaningful.\n"
        + SIMPLE_WORD_RULES +
        extra +
        f"\nROLE FOR A: {a_role_hint}\n"
        f"RELATION (for context): {relation_text}\n"
    )

def _prompt_generate_B(A: str, relation_text: str) -> str:
    return (
        "You are given A and a relation description. Produce ONE English word B such that (A, B) fits the relation.\n"
        + SIMPLE_WORD_RULES +
        f"\nRELATION: {relation_text}\nA: {A}\n"
    )

def _prompt_generate_C(a_role_hint: str, relation_text: str, avoid: str) -> str:
    return (
        "Pick a word C that plays the SAME ROLE as A would for this relation and is different from the avoid list.\n"
        + SIMPLE_WORD_RULES +
        f"\nROLE (same as A): {a_role_hint}\n"
        f"RELATION (for steering only): {relation_text}\n"
        f"AVOID: {avoid}\n"
    )

def _prompt_complete_analogy(A: str, B: str, C: str) -> str:
    # Relation-blind completion
    return (
        "Complete the analogy. Produce EXACTLY one word D.\n"
        + SIMPLE_WORD_RULES +
        f"\n{A} : {B} :: {C} : ____\n"
    )

def _prompt_grade(A: str, B: str, C: str, D: str, relation_text: str) -> str:
    return (
        "Grade whether BOTH pairs (A,B) and (C,D) satisfy the relation.\n"
        "Reply ONLY:\n"
        "GRADE: Correct or Incorrect\n"
        "REASON: <short reason>\n\n"
        f"RELATION: {relation_text}\nA:{A}  B:{B}  C:{C}  D:{D}\n"
    )

# ---- Helpers ----
def _one_word(s: str) -> str:
    w = clean_answer(s)
    return w if _SINGLE_TOKEN.match(w) else ""

def _ask_one(model: str, prompt: str, tries: int = 3) -> Tuple[str, str]:
    raw = ""
    for _ in range(tries):
        raw = generate_inference(prompt, model)
        parsed = _one_word(raw)
        if parsed:
            return parsed, raw
    return (parsed if 'parsed' in locals() else "", raw)

def _fallback_C(a_role: str, avoid: set[str]) -> str:
    # Tiny role-aware fallback for C
    if "location" in a_role or "time" in a_role:
        pool = ["winter", "summer", "library", "market"]
    elif "class" in a_role or "category" in a_role:
        pool = ["animal", "vehicle", "instrument", "fruit"]
    elif "whole object" in a_role or "collection" in a_role:
        pool = ["car", "house", "forest", "team"]
    elif "opposite" in a_role:
        pool = ["hot", "buy", "enter", "rise"]
    elif "agent" in a_role or "action" in a_role:
        pool = ["painter", "soldier", "hammer", "cut"]
    elif "sign" in a_role or "symbol" in a_role or "record" in a_role:
        pool = ["map", "flag", "siren", "diary"]
    elif "cause" in a_role or "effect" in a_role or "purpose" in a_role:
        pool = ["hunger", "rain", "study", "joke"]
    else:
        pool = ["object", "entity", "item"]
    for w in pool:
        if w not in avoid:
            return w
    return "item"

# ---- Public entrypoint ----
def run_trial(model: str, *, verbose: bool = True) -> Dict:
    """
    Build a pair analogy by:
      1) generating A under a role hint tied to the chosen relation,
      2) generating B so (A,B) satisfies the relation,
      3) generating C with the same role as A (different token),
      4) completing D in A:B :: C:D *without* revealing the relation,
      5) grading both pairs under the relation.

    Returns a dict with keys compatible with the runner + infer_hit (uses 'grade_correct').
    """
    rel_key, rel_name, rel_text, a_role = RNG.choice(RELATIONS)
    if verbose:
        print(f"\n[relation] {rel_key} — {rel_name}")

    # # ---- Step 1: A (relation-conditioned)
    # pA = _prompt_generate_A(rel_key, rel_text, a_role)
    # A, rawA = _ask_one(model, pA)
    # if not A:
    #     # Simple safe fallback per role
    #     A = _fallback_C(a_role, set())  # reuse fallback as a generic role-compatible generator
    # if verbose:
    #     print(f"[A] prompt=>\n{pA}\n[A] raw={rawA.strip()}\n[A] parsed={A}")
    # ---- Step 1: A (dictionary resampling + role validation)
    A, pA, rawA = _pick_from_dict_for_role(model, a_role, rel_text, avoid=set())
    if verbose:
        print(f"[A] validate-prompt=>\n{pA}\n[A] raw={rawA.strip()}\n[A] chosen={A}")


    # ---- Step 2: B (relation-aware)
    pB = _prompt_generate_B(A, rel_text)
    B, rawB = _ask_one(model, pB)
    if not B:
        B = "item"
    if verbose:
        print(f"[B] prompt=>\n{pB}\n[B] raw={rawB.strip()}\n[B] parsed={B}")

    # # ---- Step 3: C (same role as A; simple retry + tiny fallback)
    # pC = _prompt_generate_C(a_role_hint=a_role, relation_text=rel_text, avoid=A)
    # C, rawC = _ask_one(model, pC)
    # if (not C) or (C == A):
    #     pC = _prompt_generate_C(a_role_hint=a_role, relation_text=rel_text, avoid=f"{A}, {B}")
    #     C, rawC = _ask_one(model, pC)
    # if (not C) or (C == A):
    #     C = _fallback_C(a_role, {A, B})
    # if verbose:
    #     print(f"[C] prompt=>\n{pC}\n[C] raw={rawC.strip()}\n[C] parsed={C}")
    # ---- Step 3: C (same role; dictionary resampling; ensure C != A)
    C, pC, rawC = _pick_from_dict_for_role(model, a_role, rel_text, avoid={A})
    if verbose:
        print(f"[C] validate-prompt=>\n{pC}\n[C] raw={rawC.strip()}\n[C] chosen={C}")


    # ---- Step 4: D (relation-blind completion)
    pD = _prompt_complete_analogy(A, B, C)
    D, rawD = _ask_one(model, pD)
    if not D:
        D = "item"
    if verbose:
        print(f"[D] prompt=>\n{pD}\n[D] raw={rawD.strip()}\n[D] parsed={D}")

    # ---- Step 5: grade both pairs under the relation
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
        "prompts": {"A": pA, "B": pB, "C": pC, "D": pD, "grade": pG},
        "raw": {"A": rawA, "B": rawB, "C": rawC, "D": rawD, "grade": rawG},
        "grade_correct": is_correct,
    }
