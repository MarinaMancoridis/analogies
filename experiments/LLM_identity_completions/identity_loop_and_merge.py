from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    default_triples_path = (
        here.parent.parent / "static_triples" / "identity" / "identity_triples.json"
    )
    default_out_dir = here / "qualtrics_loop_and_merge"

    parser = argparse.ArgumentParser(
        description=(
            "Create Qualtrics Loop & Merge files from identity triples. "
            "Outputs one file per survey chunk with columns A,B,C."
        )
    )
    parser.add_argument(
        "--triples-path",
        type=Path,
        default=default_triples_path,
        help="Path to identity_triples.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_out_dir,
        help="Output directory for chunked Qualtrics files",
    )
    parser.add_argument(
        "--num-surveys",
        type=int,
        default=5,
        help="Number of output survey files",
    )
    parser.add_argument(
        "--rows-per-survey",
        type=int,
        default=20,
        help="Rows (triples) per survey file",
    )
    return parser.parse_args()


def load_triples(path: Path) -> List[Dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    triples = payload.get("triples")
    if not isinstance(triples, list):
        raise ValueError(f"Expected a list under 'triples' in {path}")
    return triples


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["A", "B", "C"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    triples = load_triples(args.triples_path)

    total_needed = args.num_surveys * args.rows_per_survey
    if len(triples) < total_needed:
        raise ValueError(
            f"Not enough triples ({len(triples)}) for requested split ({total_needed})."
        )

    # Keep canonical order by triple_id (1..100), then split into consecutive chunks.
    sorted_triples = sorted(triples, key=lambda t: int(t["triple_id"]))
    rows = [{"A": t["A"], "B": t["B"], "C": t["C"]} for t in sorted_triples[:total_needed]]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(args.num_surveys):
        start = i * args.rows_per_survey
        end = start + args.rows_per_survey
        out_path = args.out_dir / f"identity_loop_and_merge_survey_{i + 1}.csv"
        write_csv(out_path, rows[start:end])
        print(f"Wrote {end - start} rows: {out_path}")


if __name__ == "__main__":
    main()
