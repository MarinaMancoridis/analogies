from __future__ import annotations

import datetime as dt
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


SEED = 12345
N_PER_RELATION = 50

ROOT = Path(__file__).resolve().parents[2]
RELATIONAL_DIR = ROOT / "analogy_types" / "relational"
OUT_DIR = ROOT / "static_triples" / "relational"
OUT_PATH_JSON = OUT_DIR / "relation_triples.json"


def _utc_now() -> str:
    return dt.datetime.utcnow().isoformat() + "Z"


def _discover_relation_dirs() -> List[Path]:
    relation_dirs: List[Path] = []
    for child in sorted(RELATIONAL_DIR.iterdir()):
        if not child.is_dir():
            continue
        if (child / "curated_words.json").exists():
            relation_dirs.append(child)
    if len(relation_dirs) != 10:
        raise ValueError(f"Expected 10 relation folders, found {len(relation_dirs)}")
    return relation_dirs


def _load_curated_words(path: Path) -> List[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    words = payload.get("words", [])
    words = [w.strip().lower() for w in words if isinstance(w, str) and w.strip()]
    if len(words) < 2:
        raise ValueError(f"Need at least 2 words in {path}, found {len(words)}")
    return words


def _sample_pairs(words: List[str], n_pairs: int, rng: random.Random) -> List[Tuple[str, str]]:
    seen = set()
    pairs: List[Tuple[str, str]] = []
    while len(pairs) < n_pairs:
        a, c = rng.sample(words, 2)
        key = (a, c)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return pairs


def build_relation_triples(n_per_relation: int = N_PER_RELATION, seed: int = SEED) -> Dict[str, Any]:
    rng = random.Random(seed)
    triples: List[Dict[str, Any]] = []
    triple_id = 1

    for relation_dir in _discover_relation_dirs():
        relation_key = relation_dir.name
        curated_path = relation_dir / "curated_words.json"
        words = _load_curated_words(curated_path)
        pairs = _sample_pairs(words, n_per_relation, rng)
        for relation_triple_id, (a, c) in enumerate(pairs, start=1):
            triples.append(
                {
                    "triple_id": triple_id,
                    "relation_key": relation_key,
                    "relation_triple_id": relation_triple_id,
                    "A": a,
                    "C": c,
                    "curated_words_path": str(curated_path),
                }
            )
            triple_id += 1

    return {
        "created_at_utc": _utc_now(),
        "seed": seed,
        "n_relations": 10,
        "triples_per_relation": n_per_relation,
        "n_triples": len(triples),
        "selection_pipeline": (
            "For each relation, sample unique ordered (A,C) pairs "
            "from that relation's curated_words.json."
        ),
        "triples": triples,
    }


def main() -> None:
    payload = build_relation_triples()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {payload['n_triples']} triples to {OUT_PATH_JSON}")


if __name__ == "__main__":
    main()
