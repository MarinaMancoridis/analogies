from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

import nltk
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency

from analogies.common import ALL, _brys_score, ac_label_brys

_SINGLE_TOKEN = re.compile(r"^[a-z]+$")

_RANK_INDEX: Optional[Dict[str, int]] = None


def _ensure_rank_index() -> None:
    global _RANK_INDEX
    if _RANK_INDEX is not None:
        return
    _RANK_INDEX = {w.lower(): i + 1 for i, w in enumerate(ALL)}


def word_metrics(word: str) -> Dict:
    w = word.lower().strip()
    _ensure_rank_index()

    try:
        pos_tag = nltk.pos_tag([w], tagset="universal")[0][1]
    except Exception:
        pos_tag = None

    try:
        noun_synsets = wn.synsets(w, pos=wn.NOUN)
    except Exception:
        noun_synsets = []

    try:
        all_synsets = wn.synsets(w)
    except Exception:
        all_synsets = []

    try:
        brys = _brys_score(w)
        brys_label = ac_label_brys(w)
    except Exception:
        brys = None
        brys_label = None

    try:
        zipf = float(zipf_frequency(w, "en"))
    except Exception:
        zipf = None

    hyponym_counts = []
    lexnames = []

    for s in noun_synsets:
        try:
            hyponym_counts.append(len(s.hyponyms()))
        except Exception:
            hyponym_counts.append(0)
        try:
            lexnames.append(s.lexname())
        except Exception:
            pass

    return {
        "word": w,
        "len": len(w),
        "pos": pos_tag,
        "pop_rank": _RANK_INDEX.get(w),
        "pop_zipf": zipf,
        "polysemy": len(all_synsets),
        "noun_synset_count": len(noun_synsets),
        "max_hyponyms": max(hyponym_counts) if hyponym_counts else 0,
        "mean_hyponyms": (sum(hyponym_counts) / len(hyponym_counts)) if hyponym_counts else 0.0,
        "lexnames": sorted(set(lexnames)),
        "brys": brys,
        "brys_label": brys_label,
    }


def is_candidate_noun(word: str) -> bool:
    if not isinstance(word, str):
        return False

    w = word.lower().strip()

    if not _SINGLE_TOKEN.match(w):
        return False

    m = word_metrics(w)

    if m["pos"] != "NOUN":
        return False

    if m["pop_zipf"] is None:
        return False

    if m["pop_zipf"] < 2.7 or m["pop_zipf"] > 6.0:
        return False

    if m["noun_synset_count"] == 0:
        return False

    return True


def candidate_score(m: Dict) -> float:
    score = 0.0

    zipf = m.get("pop_zipf")
    if zipf:
        score += max(0, 2.5 - abs(zipf - 4.2))

    score += min(m.get("noun_synset_count", 0), 6) * 0.25
    score += min(m.get("max_hyponyms", 0), 20) * 0.1

    poly = m.get("polysemy", 0)
    if poly > 10:
        score -= 0.5

    return score


def generate_candidates(sample_size: int = 1000, seed: int = 123) -> List[Dict]:
    rng = random.Random(seed)

    pool = [w for w in ALL if is_candidate_noun(w)]
    sample = rng.sample(pool, min(sample_size, len(pool)))

    rows = []
    for w in sample:
        m = word_metrics(w)
        m["score"] = candidate_score(m)
        rows.append(m)

    rows.sort(key=lambda x: (-x["score"], x["word"]))
    return rows


def main():
    rows = generate_candidates()

    here = Path(__file__).parent
    out_path = here / "class_inclusion_candidates.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(rows)} candidates to {out_path}")

    print("\nTop 20 preview:\n")
    for r in rows[:20]:
        print({
            "word": r["word"],
            "score": round(r["score"], 3),
            "zipf": r["pop_zipf"],
            "max_hyponyms": r["max_hyponyms"],
            "polysemy": r["polysemy"],
            "lexnames": r["lexnames"][:3],
        })


if __name__ == "__main__":
    main()