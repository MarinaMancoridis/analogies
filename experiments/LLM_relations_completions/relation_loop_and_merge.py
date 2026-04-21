from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    default_triples_path = (
        here.parent.parent / "static_triples" / "relational" / "relation_triples.json"
    )
    default_out_dir = here / "qualtrics_loop_and_merge"
    default_out_path = default_out_dir / "relation_loop_and_merge_all.csv"

    parser = argparse.ArgumentParser(
        description=(
            "Create a single Qualtrics Loop & Merge CSV from relational triples. "
            "Outputs one file with triple_id, relation_key, A, B (blank), C."
        )
    )
    parser.add_argument(
        "--triples-path",
        type=Path,
        default=default_triples_path,
        help="Path to relation_triples.json",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=default_out_path,
        help="Output CSV path",
    )
    return parser.parse_args()


def load_triples(path: Path) -> List[Dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    triples = payload.get("triples")
    if not isinstance(triples, list):
        raise ValueError(f"Expected a list under 'triples' in {path}")
    return triples


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["triple_id", "relation_key", "A", "B", "C"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    triples = load_triples(args.triples_path)

    sorted_triples = sorted(triples, key=lambda t: int(t["triple_id"]))
    rows = [
        {
            "triple_id": str(t["triple_id"]),
            "relation_key": str(t["relation_key"]),
            "A": str(t["A"]),
            "B": "",
            "C": str(t["C"]),
        }
        for t in sorted_triples
    ]

    write_csv(args.out_path, rows)
    print(f"Wrote {len(rows)} rows: {args.out_path}")


if __name__ == "__main__":
    main()
