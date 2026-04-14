from __future__ import annotations

import datetime
import json
import os
import random
from typing import Any, Dict, List

from analogies import common as common_mod
from analogies.experiments.identity_10_models_with_metadata import word_metrics

N_TRIPLES = 100
SEED = 12345
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
OUT_DIR = os.path.join(ROOT, "static_triples", "identity")
OUT_PATH_NO_EXT = os.path.join(OUT_DIR, "identity_triples")
OUT_PATH_JSON = os.path.join(OUT_DIR, "identity_triples.json")
BRYS_PATH = os.path.join(ROOT, "concreteness.txt")


def _utc_now() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def _is_complete_metrics(metrics: Dict[str, Any]) -> bool:
    required_fields = ["len", "pop_rank", "pop_zipf", "polysemy", "brys", "brys_label", "pos"]
    return all(metrics.get(field) is not None for field in required_fields)


def _sample_valid_word(rng: random.Random, max_tries: int = 20000) -> tuple[str, Dict[str, Any], int]:
    tries = 0
    while tries < max_tries:
        tries += 1
        w = rng.choice(common_mod.ALL)
        m = word_metrics(w)
        if _is_complete_metrics(m):
            return w, m, tries
    raise RuntimeError(f"Could not sample a word with complete metrics in {max_tries} tries.")


def build_identity_triples(n: int = N_TRIPLES, seed: int = SEED) -> Dict[str, Any]:
    rng = random.Random(seed)
    common_mod.load_brysbaert_norms(BRYS_PATH)

    triples: List[Dict[str, Any]] = []
    used_a: set[str] = set()
    attempts_total = 0

    while len(triples) < n:
        a_word, a_metrics, a_tries = _sample_valid_word(rng)
        attempts_total += a_tries
        if a_word in used_a:
            continue

        c_word, c_metrics, c_tries = _sample_valid_word(rng)
        attempts_total += c_tries
        while c_word == a_word:
            c_word, c_metrics, c_tries = _sample_valid_word(rng)
            attempts_total += c_tries

        triple_id = len(triples) + 1
        triples.append(
            {
                "triple_id": triple_id,
                "analogy_type": "identity",
                "relation": "A equals B",
                "A": a_word,
                "B": a_word,
                "C": c_word,
                "A_metrics": a_metrics,
                "B_metrics": dict(a_metrics),
                "C_metrics": c_metrics,
            }
        )
        used_a.add(a_word)

    return {
        "created_at_utc": _utc_now(),
        "seed": seed,
        "n_triples": len(triples),
        "source_vocab": "common_mod.ALL (top 50k via wordfreq)",
        "selection_pipeline": "Randomly sample from ALL; retry until all metrics fields are populated.",
        "total_sampling_attempts": attempts_total,
        "triples": triples,
    }


def main() -> None:
    payload = build_identity_triples()
    os.makedirs(OUT_DIR, exist_ok=True)
    serialized = json.dumps(payload, indent=2, ensure_ascii=False)
    with open(OUT_PATH_NO_EXT, "w", encoding="utf-8") as f:
        f.write(serialized)
    with open(OUT_PATH_JSON, "w", encoding="utf-8") as f:
        f.write(serialized)
    print(f"Wrote {payload['n_triples']} triples to {OUT_PATH_NO_EXT} and {OUT_PATH_JSON}")


if __name__ == "__main__":
    main()
