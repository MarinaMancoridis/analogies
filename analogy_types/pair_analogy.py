# pair_analogy.py
import os, json, re, random, uuid, datetime
from typing import Dict, Tuple, Optional
from wordfreq import top_n_list
from analogies.utils import generate_inference

RNG = random.Random()
WORDS = top_n_list("en", 50000)

# ---- Relation catalog (concise, implementation-oriented) ----
# Each item: (key, human_name, relation_instruction)
RELATIONS = [
    ("class_taxonomic",      "Class-Inclusion: Taxonomic (Class:Individual)",
     "B is a specific instance or subtype of A (e.g., flower:tulip)."),
    ("class_individual",     "Class-Inclusion: Class:Individual",
     "B is a well-known individual or named example of class A (avoid proper nouns unless obvious)."),
    ("part_component",       "Part-Whole: Object:Component",
     "B is a physical component/part of A (e.g., car:engine)."),
    ("part_collection",      "Part-Whole: Collection:Member",
     "B is a typical member of collection A (e.g., forest:tree)."),
    ("similar_synonym",      "Similar: Synonymy",
     "B is a close synonym of A (e.g., car:auto)."),
    ("similar_dimensional",  "Similar: Dimensional Similarity",
     "B is similar to A along a graded dimension (e.g., simmer:boil)."),
    ("contrast_contrary",    "Contrast: Contrary",
     "B is the opposite of A (e.g., old:young)."),
    ("contrast_reverse",     "Contrast: Reverse",
     "B is the reverse action/state of A (e.g., buy:sell)."),
    ("attribute_item_attr",  "Attribute: Item:Attribute",
     "B is a typical attribute/quality of A (e.g., beggar:poor)."),
    ("attribute_obj_state",  "Attribute: Object:State",
     "B is a state commonly experienced by A (e.g., coward:fear)."),
    ("nonattr_item",         "Non-Attribute: Item:Nonattribute",
     "B is an attribute that A typically lacks (e.g., fire:cold)."),
    ("nonattr_object",       "Non-Attribute: Object:Nonstate",
     "B is a state incompatible with A (e.g., corpse:life)."),
    ("case_agent_instr",     "Case Relations: Agent:Instrument",
     "B is a typical instrument used by agent A (e.g., soldier:gun)."),
    ("case_action_object",   "Case Relations: Action:Object",
     "B is a typical object or patient of action A (e.g., plow:earth)."),
    ("cause_effect",         "Cause-Purpose: Cause:Effect",
     "B is the typical effect caused by A (e.g., joke:laughter)."),
    ("cause_compensate",     "Cause-Purpose: Cause:Compensatory action",
     "B is an action that counters or remedies A (e.g., hunger:eat)."),
    ("spacetime_location",   "Space-Time: Location:Item",
     "B is a typical item found at location A (e.g., library:book)."),
    ("spacetime_time_item",  "Space-Time: Time:Associated Item",
     "B is typically associated with time/season A (e.g., winter:snow)."),
    ("reference_sign",       "Reference: Sign:Significant",
     "A is a sign that indicates B (e.g., siren:danger)."),
    ("reference_repr",       "Reference: Representation",
     "A is a representation/record of B (e.g., diary:person)."),
]

# ---- Small utils ----
def _is_simple_token(w: str) -> bool:
    return w.isascii() and w.isalpha()

def _clean_one_word(resp: str) -> str:
    # Prefer "ANSWER: <word>"
    m = re.search(r"ANSWER:\s*([A-Za-z\-']+)", resp, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    m2 = re.search(r"\b([A-Za-z\-']+)\b", resp)
    return m2.group(1).lower() if m2 else ""

def sample_word() -> str:
    while True:
        w = RNG.choice(WORDS).lower()
        if _is_simple_token(w):
            return w

# ---- Concept check (same contract as your existing code) ----
def is_concept(word: str, model: str) -> Tuple[bool, Optional[str], str]:
    prompt = (
        "Determine if the following English WORD is a concept "
        "(noun/verb/adjective/adverb), not a function word "
        "(conjunction/preposition/article/pronoun), nor a proper noun.\n"
        "Respond ONLY:\nANSWER: Yes or No\nCONCEPT: <canonical form>\n\n"
        f"{word}\n"
    )
    resp = generate_inference(prompt, model)
    yes = bool(re.search(r"ANSWER:\s*Yes\b", resp, re.IGNORECASE))
    c = re.search(r"CONCEPT:\s*(.*)", resp, re.IGNORECASE)
    canon = c.group(1).strip().lower() if c else None
    return yes, canon, resp

def sample_concept(model: str) -> str:
    while True:
        w = sample_word()
        ok, canon, raw = is_concept(w, model)
        print(f"[is_concept] w={w} ok={ok} canon={canon} raw={raw.strip()[:120]}")
        if ok:
            return canon or w

# ---- Prompts ----
def prompt_generate_B(A: str, relation_text: str) -> str:
    return (
        "You are given a relation description. Produce ONE English word B such that (A, B) fits the relation.\n"
        "Output ONLY: ANSWER: <word>\n\n"
        f"RELATION: {relation_text}\nA: {A}\n"
    )

def prompt_complete_analogy(A: str, B: str, C: str, relation_text: str) -> str:
    return (
        "Complete the analogy consistent with the relation described. "
        "Output ONLY: ANSWER: <word>\n\n"
        f"RELATION: {relation_text}\n"
        f"{A} : {B} :: {C} : ____\n"
    )

def prompt_grade(A: str, B: str, C: str, D: str, relation_text: str) -> str:
    return (
        "Grade whether BOTH pairs (A,B) and (C,D) satisfy the relation. "
        "Reply ONLY:\nGRADE: Correct or Incorrect\n"
        "REASON: <short reason>\n\n"
        f"RELATION: {relation_text}\nA:{A}  B:{B}  C:{C}  D:{D}\n"
    )

# ---- Core trial ----
def run_single_pair_trial(model: str) -> Dict:
    rel_key, rel_name, rel_text = RNG.choice(RELATIONS)
    print(f"\n[relation] {rel_key} — {rel_name}")

    A = sample_concept(model)
    print(f"[A] {A}")

    # Step 3: ask LLM for B
    pB = prompt_generate_B(A, rel_text)
    rawB = generate_inference(pB, model)
    B = _clean_one_word(rawB)
    print(f"[B] prompt=>\n{pB}\n[B] raw={rawB.strip()}\n[B] parsed={B}")

    # Step 4: sample C
    C = sample_concept(model)
    while C == A:
        C = sample_concept(model)
    print(f"[C] {C}")

    # Step 5: ask LLM to fill D
    pD = prompt_complete_analogy(A, B, C, rel_text)
    rawD = generate_inference(pD, model)
    D = _clean_one_word(rawD)
    print(f"[D] prompt=>\n{pD}\n[D] raw={rawD.strip()}\n[D] parsed={D}")

    # Step 6: grade
    pG = prompt_grade(A, B, C, D, rel_text)
    rawG = generate_inference(pG, model)
    is_correct = bool(re.search(r"GRADE:\s*Correct\b", rawG, re.IGNORECASE))
    print(f"[grade] prompt=>\n{pG}\n[grade] raw={rawG.strip()}\n[grade] correct={is_correct}")

    return {
        "model": model,
        "relation_key": rel_key,
        "relation_name": rel_name,
        "relation_text": rel_text,
        "A": A, "B": B, "C": C, "D": D,
        "prompts": {"B": pB, "D": pD, "grade": pG},
        "raw": {"B": rawB, "D": rawD, "grade": rawG},
        "grade_correct": is_correct,
    }

# ---- Runner ----
def run_pair_analogy_trials(model: str, n_trials: int, out_dir: Optional[str] = None) -> Dict:
    """
    Runs n_trials of the pair-analogy pipeline and writes JSON logs.
    If out_dir is None, creates responses/<run_id>/pairs/.
    Returns summary dict.
    """
    if out_dir is None:
        run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        out_dir = os.path.join("responses", run_id, "pairs")
    os.makedirs(out_dir, exist_ok=True)

    trials = []
    hits = 0
    for i in range(n_trials):
        print(f"\n===== Pair Trial {i+1}/{n_trials} =====")
        t = run_single_pair_trial(model)
        hits += int(t["grade_correct"])
        trials.append(t)

    summary = {
        "model": model,
        "n_trials": n_trials,
        "correct": hits,
        "accuracy": hits / max(1, n_trials),
    }

    with open(os.path.join(out_dir, "pair_trials.json"), "w") as f:
        json.dump(trials, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "pair_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== Pair Analogy Run Complete ===")
    print(json.dumps(summary, indent=2))
    print(f"Saved to {out_dir}")
    return summary

# Optional CLI shim
if __name__ == "__main__":
    run_pair_analogy_trials(model="gpt-5", n_trials=10)
