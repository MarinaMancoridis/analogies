"""
Colon-format *identity* analogies:

  A : A :: C : ____

- **A** is always one of the palindrome / catchy pool (8 words).
- **C** is sampled like static identity triples: uniform from `common.ALL`
  (top-50k wordfreq), retry until all lexical metrics fields are populated —
  same pipeline as `static_triples/identity/generate_identity_triples.py`.

Builds **56** unique (A, C) analogies (no pair repeats): each catchy A gets
**7** distinct C words. All **10** models run on the same list → **560** API calls.

On stdout: ✓ (i/N) on correct; [WRONG] / [ERROR] on failure.

  cd experiments/catchy_example && python catchy_example_finder.py
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from analogies import common as common_mod
from analogies.analogy_types.structural.identity import word_metrics
from analogies.utils import generate_inference

CATCHY_WORDS: List[str] = [
    "madam",
    "civic",
    "kayak",
    "level",
    "noon",
    "rotator",
    "radar",
    "racecar",
]

N_C_PER_A = 7
N_ANALOGIES = len(CATCHY_WORDS) * N_C_PER_A  # 56

MODELS: List[str] = [
    "meta-llama/llama-3.3-70b-instruct",
    "gpt-4o",
    "openai/gpt-4.1",
    "openai/gpt-5.4-mini",
    "google/gemini-3-flash-preview",
    "google/gemini-3.1-flash-lite-preview",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-opus-4.6",
    "deepseek-ai/DeepSeek-V3",
    "deepseek/deepseek-v3.2",
]

BRYS_PATH = REPO_ROOT / "concreteness.txt"


def _is_complete_metrics(metrics: Dict[str, Any]) -> bool:
    required = ["len", "pop_rank", "pop_zipf", "polysemy", "brys", "brys_label", "pos"]
    return all(metrics.get(f) is not None for f in required)


def _sample_valid_c(rng: random.Random, *, avoid: Set[str], max_tries: int = 50_000) -> str:
    """One C word: same validity gate as `generate_identity_triples._sample_valid_word`."""
    avoid_l = {x.lower() for x in avoid}
    tries = 0
    while tries < max_tries:
        tries += 1
        w = rng.choice(common_mod.ALL)
        wl = w.lower()
        if wl in avoid_l:
            continue
        try:
            m = word_metrics(w)
        except Exception:
            continue
        if _is_complete_metrics(m):
            return wl
    raise RuntimeError(f"Could not sample a valid C word after {max_tries} tries (avoid={avoid_l!r}).")


def build_analogies(seed: int) -> List[Tuple[str, str]]:
    """
    Return 56 (A, C) with A from CATCHY_WORDS, C from ALL/metrics pipeline,
    seven distinct C per A, C != A, and no duplicate (A,C) rows.
    """
    try:
        common_mod.load_brysbaert_norms(str(BRYS_PATH))
    except Exception:
        pass

    rng = random.Random(seed)
    seen_pairs: Set[Tuple[str, str]] = set()
    pairs: List[Tuple[str, str]] = []

    for a in CATCHY_WORDS:
        a_l = a.lower()
        c_for_this_a: Set[str] = set()
        while len(c_for_this_a) < N_C_PER_A:
            avoid = {a_l} | c_for_this_a
            c = _sample_valid_c(rng, avoid=avoid)
            key = (a_l, c)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            c_for_this_a.add(c)
            pairs.append((a, c))

    if len(pairs) != N_ANALOGIES or len(seen_pairs) != N_ANALOGIES:
        raise RuntimeError(f"internal: expected {N_ANALOGIES} unique pairs, got {len(pairs)}")

    return pairs


def _colon_identity_prompt(a: str, c: str) -> str:
    return (
        "Complete the analogy. Reply ONLY as: ANSWER: <word>\n\n"
        f"{a} : {a} :: {c} : ____"
    )


def _clean_answer(resp: str) -> str:
    text = (resp or "").strip()
    m = re.search(r"ANSWER:\s*([A-Za-z][A-Za-z'-]*)", text, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    m2 = re.search(r"\b([A-Za-z][A-Za-z'-]*)\b", text)
    return m2.group(1).lower() if m2 else ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--shuffle",
        action="store_true",
        help="Randomize analogy order before running (uses --seed).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for sampling C words and for --shuffle.",
    )
    ap.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Subset of models (default: all 10).",
    )
    args = ap.parse_args()

    models = args.models if args.models else MODELS
    pairs = build_analogies(args.seed)
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(pairs)

    n_jobs = len(pairs) * len(models)
    print(
        f"Catchy-A + ALL-vocab-C identity colon | {len(pairs)} analogies × {len(models)} models = {n_jobs} API calls",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"A pool: {', '.join(CATCHY_WORDS)}\n"
        f"C sampling: same as static_triples/identity/generate_identity_triples.py\n"
        f"--- stdout: ✓ (i/{n_jobs}) on correct; [WRONG]/[ERROR] on failure ---\n",
        file=sys.stderr,
        flush=True,
    )

    n_ok = 0
    n_bad = 0
    n_err = 0
    done = 0

    for a, c in pairs:
        prompt = _colon_identity_prompt(a, c)
        expected = c.lower()
        for model in models:
            done += 1
            try:
                raw = generate_inference(prompt, model)
                parsed = _clean_answer(raw)
                ok = parsed == expected
            except Exception as e:
                n_err += 1
                print(
                    f"[ERROR] model={model!r}  |  {a!r} : {a!r} :: {c!r} : ?  |  expected {c!r}  |  {e!r}",
                    flush=True,
                )
                continue
            if ok:
                n_ok += 1
                print(f"✓ ({done}/{n_jobs})", flush=True)
            else:
                n_bad += 1
                raw_short = (raw or "").replace("\n", " ").strip()
                if len(raw_short) > 120:
                    raw_short = raw_short[:117] + "..."
                print(
                    f"[WRONG] model={model!r}  |  {a!r} : {a!r} :: {c!r} : ?  |  "
                    f"expected {c!r}  |  model said {parsed!r}  |  raw: {raw_short!r}",
                    flush=True,
                )
            if done % 50 == 0:
                print(f"... {done}/{n_jobs}", file=sys.stderr, flush=True)

    print(
        f"\n--- Summary ---\n"
        f"Correct: {n_ok}  |  Wrong: {n_bad}  |  API errors: {n_err}  |  Total: {n_ok + n_bad + n_err}",
        file=sys.stderr,
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
