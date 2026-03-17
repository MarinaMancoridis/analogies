from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import nltk
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency

from analogies.utils import generate_inference
from analogies.common import clean_answer, ALL, _brys_score, ac_label_brys

TASK_NAME = "relational_similar"

REL_KEY = "similar"
REL_NAME = "Similar"
REL_TEXT = "B is similar or nearly equivalent to A (synonym or close along a dimension). e.g., car:auto; simmer:boil."
A_ROLE_HINT = "a common word that has a close synonym"

PROMPT_VARIATIONS = {
    "colon": lambda A, B, C: (
        f"Complete the analogy. Reply ONLY as: ANSWER: <word>\n\n"
        f"{A} : {B} :: {C} : ____"
    ),
    "english": lambda A, B, C: (
        f"Complete the analogy. Reply ONLY as: ANSWER: <word>\n\n"
        f"{A} is to {B} as {C} is to ____"
    ),
    "one_shot": lambda A, B, C: (
        f"Complete the analogy. Reply ONLY as: ANSWER: <word>\n\n"
        f"{A} : {B} :: {C} : ____\n\n"
        f"Here is one example that follows the same relation:\n"
        f"car : auto :: rapid : fast"
    ),
    "few_shot": lambda A, B, C: (
        f"Complete the analogy. Reply ONLY as: ANSWER: <word>\n\n"
        f"{A} : {B} :: {C} : ____\n\n"
        f"Here are some examples that follow the same relation:\n"
        f"car : auto :: rapid : fast\n"
        f"silent : quiet :: angry : mad"
    ),
}

_SINGLE_TOKEN = re.compile(r"^[a-z]+$")
_RANK_INDEX: Optional[Dict[str, int]] = None


def _load_curated_words() -> List[str]:
    path = Path(__file__).parent / "curated_words.json"
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    words = obj.get("words", [])
    words = [w.strip().lower() for w in words if isinstance(w, str) and w.strip()]
    if len(words) < 2:
        raise ValueError(f"Need at least 2 curated words in {path}, found {len(words)}")
    return words


def _one_word(s: str) -> str:
    w = clean_answer(s)
    return w if _SINGLE_TOKEN.match(w) else ""


def _ask_one(model: str, prompt: str, tries: int = 3) -> Tuple[str, str]:
    raw = ""
    parsed = ""
    for _ in range(tries):
        raw = generate_inference(prompt, model)
        parsed = _one_word(raw)
        if parsed:
            return parsed, raw
    return parsed, raw


def _ensure_rank_index() -> None:
    global _RANK_INDEX
    if _RANK_INDEX is not None:
        return
    _RANK_INDEX = {w.lower(): i + 1 for i, w in enumerate(ALL)}


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


_SIMPLE_WORD_RULES = (
    "Rules:\n"
    "- Output exactly one lowercase English word.\n"
    "- No punctuation, no spaces, no hyphens, no numbers, no proper nouns.\n"
    "- The answer must be a close synonym or near-equivalent of A, not merely associated with A.\n"
    "Output ONLY: ANSWER: <word>\n"
)


def _prompt_generate_B(A: str) -> str:
    return (
        "You are given a word A and a relation description. "
        "Produce ONE English word B such that (A, B) fits the relation.\n"
        + _SIMPLE_WORD_RULES +
        "\n"
        f"RELATION: {REL_TEXT}\n"
        f"ROLE FOR A: {A_ROLE_HINT}\n"
        f"A: {A}\n"
    )


def _prompt_grade(A: str, B: str, C: str, D: str) -> str:
    return (
        "Grade whether BOTH pairs (A,B) and (C,D) satisfy the relation.\n"
        "For this relation, B should be similar to or nearly equivalent to A, and D should be similar to or nearly equivalent to C. "
        "Mere association, opposition, part-whole, or nearby topic does not count.\n\n"
        "Reply ONLY:\n"
        "GRADE: Correct or Incorrect\n"
        "REASON: <short reason>\n\n"
        f"RELATION: {REL_TEXT}\n"
        f"A:{A}  B:{B}  C:{C}  D:{D}\n"
    )


def _parse_grade_label(raw: str) -> Optional[str]:
    if re.search(r"\bGRADE:\s*Correct\b", raw, re.IGNORECASE):
        return "Correct"
    if re.search(r"\bGRADE:\s*Incorrect\b", raw, re.IGNORECASE):
        return "Incorrect"
    return None


def _majority_judge(model: str, A: str, B: str, C: str, D: str, n_judges: int = 5):
    prompt = _prompt_grade(A, B, C, D)
    raws: List[str] = []
    labels: List[str] = []

    for _ in range(n_judges):
        raw = generate_inference(prompt, model)
        raws.append(raw)
        label = _parse_grade_label(raw)
        if label is not None:
            labels.append(label)

    if labels:
        counts = Counter(labels)
        majority_label = counts.most_common(1)[0][0]
        majority_correct = (majority_label == "Correct")
    else:
        counts = Counter()
        majority_label = None
        majority_correct = False

    return {
        "judge_prompt": prompt,
        "judge_raws": raws,
        "judge_labels": labels,
        "judge_counts": dict(counts),
        "judge_majority_label": majority_label,
        "judge_majority_correct": majority_correct,
        "judge_n": n_judges,
    }


def _prompt(A: str, B: str, C: str):
    return PROMPT_VARIATIONS["colon"](A, B, C)


def run_trial(model: str, *args, prompt_type: str = "colon", verbose: bool = True):
    curated_words = _load_curated_words()
    rng = random.Random()

    if len(args) >= 2:
        A, C = args[:2]
        A_mode_used = "from_dataset"
        C_mode_used = "from_dataset"
    else:
        A, C = rng.sample(curated_words, 2)
        A_mode_used = "random"
        C_mode_used = "random"

    pB = _prompt_generate_B(A)
    B, rawB = _ask_one(model, pB)
    if not B:
        B = "item"

    prompt_fn = PROMPT_VARIATIONS.get(prompt_type, _prompt)
    pD = prompt_fn(A, B, C) if callable(prompt_fn) else _prompt(A, B, C)
    D, rawD = _ask_one(model, pD)
    if not D:
        D = "item"

    judge = _majority_judge(model=model, A=A, B=B, C=C, D=D, n_judges=5)
    is_correct = judge["judge_majority_correct"]

    out = {
        "model": model,
        "analogy_type": TASK_NAME,
        "relation_key": REL_KEY,
        "relation_name": REL_NAME,
        "relation_text": REL_TEXT,
        "A_mode": A_mode_used,
        "C_mode": C_mode_used,
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "A_metrics": word_metrics(A),
        "B_metrics": word_metrics(B),
        "C_metrics": word_metrics(C),
        "D_metrics": word_metrics(D),
        "prompt": pD,
        "raw_response": rawD,
        "parsed_answer": D,
        "expected": None,
        "is_identity": is_correct,
        "prompt_type": prompt_type,
        "b_prompt": pB,
        "b_raw": rawB,
        "grade_prompt": judge["judge_prompt"],
        "grade_raw": judge["judge_raws"][-1] if judge["judge_raws"] else "",
        "grade_correct": is_correct,
        "judge_raws": judge["judge_raws"],
        "judge_labels": judge["judge_labels"],
        "judge_counts": judge["judge_counts"],
        "judge_majority_label": judge["judge_majority_label"],
        "judge_majority_correct": judge["judge_majority_correct"],
        "judge_n": judge["judge_n"],
    }

    if verbose:
        print(
            f"[relational_similar] "
            f"A={A} B={B} C={C} D={D} "
            f"judge_majority={judge['judge_majority_label']} "
            f"counts={judge['judge_counts']} "
            f"prompt={prompt_type}"
        )

    return out


def generate_dataset(n: int, rng: random.Random):
    curated_words = _load_curated_words()

    dataset = []
    rejected = 0
    attempts = 0

    while len(dataset) < n:
        attempts += 1
        A, C = rng.sample(curated_words, 2)

        if A == C:
            rejected += 1
            continue

        dataset.append((A, C))

    sampling_stats = {
        "source_vocab": "curated_words.json",
        "initial_pool_size": len(curated_words),
        "total_attempts": attempts,
        "total_rejected": rejected,
        "acceptance_rate": len(dataset) / attempts if attempts else None,
        "relation_key": REL_KEY,
        "relation_name": REL_NAME,
        "a_role_hint": A_ROLE_HINT,
        "judge_strategy": "majority_vote_over_5_attempts",
        "rule": "A and C are sampled from curated_words.json; B and D are model-generated and relation-judged by majority vote.",
    }
    return dataset, sampling_stats