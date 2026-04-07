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

TASK_NAME = "relational_reference"

REL_KEY = "reference"
REL_NAME = "Reference"
REL_TEXT = "A is a sign or representation referring to B. e.g., siren:danger; diary:person."
A_ROLE_HINT = "a sign, record, or symbol that refers to something"

PROMPT_VARIATIONS = {
    "colon": lambda A, B, C: (
        f"{A} : {B} :: {C} : ____\n\nReply ONLY: ANSWER: <word>"
    ),
    "english": lambda A, B, C: (
        f"{A} is to {B} as {C} is to ____\n\nReply ONLY: ANSWER: <word>"
    ),
    "one_shot": lambda A, B, C: (
        f"{A} : {B} :: {C} : ____\n\n"
        f"Here is one example that follows the same relation:\n"
        f"siren : danger :: diary : person\n\n"
        f"Reply ONLY: ANSWER: <word>"
    ),
    "few_shot": lambda A, B, C: (
        f"{A} : {B} :: {C} : ____\n\n"
        f"Here are some examples that follow the same relation:\n"
        f"siren : danger :: diary : person\n"
        f"map : location :: signal : warning\n\n"
        f"Reply ONLY: ANSWER: <word>"
    ),
}

_SINGLE_TOKEN = re.compile(r"^[a-z]+$")


def _load_curated_words() -> List[str]:
    path = Path(__file__).parent / "curated_words.json"
    with open(path, "r") as f:
        return json.load(f)["words"]


def _one_word(s: str) -> str:
    w = clean_answer(s)
    return w if _SINGLE_TOKEN.match(w) else ""


def _ask_one(model: str, prompt: str) -> Tuple[str, str]:
    raw = generate_inference(prompt, model)
    return _one_word(raw), raw


def _prompt_generate_B(A: str) -> str:
    return (
        f"Produce ONE word B such that A refers to B.\n"
        f"A: {A}\n"
        f"Reply ONLY: ANSWER: <word>"
    )


def _prompt_grade(A: str, B: str, C: str, D: str) -> str:
    return (
        f"Check if both pairs follow a reference relation.\n"
        f"A:{A} B:{B} C:{C} D:{D}\n\n"
        f"Reply ONLY:\nGRADE: Correct or Incorrect\nREASON: <short>"
    )


def _parse_grade(raw: str):
    if "Correct" in raw:
        return "Correct"
    if "Incorrect" in raw:
        return "Incorrect"
    return None


def _majority_judge(model, A, B, C, D, n=5):
    labels = []
    raws = []
    for _ in range(n):
        r = generate_inference(_prompt_grade(A, B, C, D), model)
        raws.append(r)
        lab = _parse_grade(r)
        if lab:
            labels.append(lab)

    counts = Counter(labels)
    maj = counts.most_common(1)[0][0] if counts else None

    return {
        "judge_raws": raws,
        "judge_counts": dict(counts),
        "judge_majority_label": maj,
        "judge_majority_correct": maj == "Correct",
    }


def run_trial(model: str, *args, prompt_type="colon"):
    words = _load_curated_words()
    rng = random.Random()

    A, C = rng.sample(words, 2)

    B, _ = _ask_one(model, _prompt_generate_B(A))
    D, _ = _ask_one(model, PROMPT_VARIATIONS[prompt_type](A, B, C))

    judge = _majority_judge(model, A, B, C, D)

    return {
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "correct": judge["judge_majority_correct"],
    }


def generate_dataset(n: int, rng: random.Random):
    words = _load_curated_words()
    return [(rng.choice(words), rng.choice(words)) for _ in range(n)], {}