from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List


def main() -> None:
    here = Path(__file__).resolve().parent
    block_files = sorted(here.glob("curate_b_block_*.csv"))

    if len(block_files) != 25:
        print(f"WARNING: Expected 25 block files, found {len(block_files)}")

    within_file_duplicates: Dict[str, List[int]] = {}
    triple_to_files: Dict[int, List[str]] = defaultdict(list)

    for block_path in block_files:
        with block_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

        ids: List[int] = []
        for row in rows:
            tid_raw = (row.get("triple_id") or "").strip()
            if not tid_raw:
                continue
            ids.append(int(tid_raw))

        counts = Counter(ids)
        dups = sorted([tid for tid, c in counts.items() if c > 1])
        if dups:
            within_file_duplicates[block_path.name] = dups

        for tid in counts.keys():
            triple_to_files[tid].append(block_path.name)

    cross_file_duplicates = {
        tid: sorted(files) for tid, files in triple_to_files.items() if len(files) > 1
    }

    print("=== Repeat Check Summary ===")
    print(f"Block files scanned: {len(block_files)}")
    print(f"Unique triple_ids seen: {len(triple_to_files)}")
    print()

    if not within_file_duplicates:
        print("No within-file triple_id duplicates found.")
    else:
        print("Within-file duplicates found:")
        for fname, dups in sorted(within_file_duplicates.items()):
            print(f"  {fname}: {dups}")
    print()

    if not cross_file_duplicates:
        print("No cross-file triple_id duplicates found.")
    else:
        print("Cross-file duplicates found:")
        for tid, files in sorted(cross_file_duplicates.items()):
            print(f"  triple_id {tid} appears in: {files}")

    print()
    if not within_file_duplicates and not cross_file_duplicates:
        print("PASS: No repeated triple_id values within or across block files.")
    else:
        print("FAIL: Repeated triple_id values detected.")


if __name__ == "__main__":
    main()
