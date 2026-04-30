from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Set


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Assign 500 triples to 75 participants x 20 items each "
            "(balanced: each triple shown exactly 3 times)."
        )
    )
    parser.add_argument(
        "--triples-csv",
        type=Path,
        default=here / "curate_b_relation_loop_and_merge_all.csv",
        help="Input triples CSV with columns: triple_id, relation_key, A, B, C.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=here / "curate_b_participant_assignments.csv",
        help="Output long-format assignment CSV.",
    )
    parser.add_argument("--participants", type=int, default=75)
    parser.add_argument("--items-per-participant", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def _load_triples(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"triple_id", "relation_key", "A", "B", "C"}
    if not rows:
        raise ValueError(f"No rows found in {path}")
    missing = required.difference(rows[0].keys())
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")
    return rows


def _build_quotas(n_participants: int) -> List[List[int]]:
    """
    3 cycles, each cycle covers all triples once.
    We need each participant to receive 20 items total:
      - baseline 6 per cycle => 18
      - +1 in two cycles => 20
    For each cycle, 50 participants get 7 and 25 get 6.
    """
    quotas: List[List[int]] = []
    for p in range(n_participants):
        low_cycle = p % 3
        row = []
        for c in range(3):
            row.append(6 if c == low_cycle else 7)
        quotas.append(row)
    return quotas


def _assign_cycle(
    *,
    triples: List[Dict[str, str]],
    cycle_idx: int,
    rng: random.Random,
    quota_remaining: List[int],
    assigned_ids_by_participant: List[Set[str]],
) -> List[int]:
    order = list(range(len(triples)))
    rng.shuffle(order)
    participant_order = list(range(len(quota_remaining)))
    rng.shuffle(participant_order)

    assigned_participant_for_item = [-1] * len(triples)

    for item_idx in order:
        triple_id = str(triples[item_idx]["triple_id"])
        candidates = [
            p
            for p in participant_order
            if quota_remaining[p] > 0 and triple_id not in assigned_ids_by_participant[p]
        ]
        if not candidates:
            raise RuntimeError(
                f"No valid participant left for triple_id={triple_id} in cycle={cycle_idx + 1}"
            )

        # Prefer participants with the most remaining slots to avoid dead-ends.
        best = max(candidates, key=lambda p: quota_remaining[p])
        assigned_participant_for_item[item_idx] = best
        quota_remaining[best] -= 1
        assigned_ids_by_participant[best].add(triple_id)

    if any(q != 0 for q in quota_remaining):
        raise RuntimeError(f"Cycle {cycle_idx + 1} did not fill all quotas: {quota_remaining}")
    return assigned_participant_for_item


def main() -> None:
    args = parse_args()
    triples = _load_triples(args.triples_csv)
    n_triples = len(triples)
    n_participants = args.participants
    k = args.items_per_participant

    total_slots = n_participants * k
    if total_slots % n_triples != 0:
        raise ValueError(
            f"participants*items_per_participant={total_slots} is not divisible by n_triples={n_triples}"
        )
    repeats = total_slots // n_triples
    if repeats != 3:
        raise ValueError(
            f"This script expects exactly 3 repeats (got {repeats}). "
            "Use 75 participants x 20 items with 500 triples."
        )

    rng = random.Random(args.seed)
    quotas = _build_quotas(n_participants)
    assigned_ids_by_participant: List[Set[str]] = [set() for _ in range(n_participants)]
    assignments_by_participant: List[List[Dict[str, str]]] = [[] for _ in range(n_participants)]

    for cycle_idx in range(3):
        quota_remaining = [quotas[p][cycle_idx] for p in range(n_participants)]
        assignees = _assign_cycle(
            triples=triples,
            cycle_idx=cycle_idx,
            rng=rng,
            quota_remaining=quota_remaining,
            assigned_ids_by_participant=assigned_ids_by_participant,
        )
        for item_idx, p in enumerate(assignees):
            row = dict(triples[item_idx])
            row["cycle"] = str(cycle_idx + 1)
            assignments_by_participant[p].append(row)

    # Final sanity checks.
    for p, rows in enumerate(assignments_by_participant):
        if len(rows) != k:
            raise RuntimeError(f"Participant {p + 1} has {len(rows)} rows, expected {k}")
        ids = [r["triple_id"] for r in rows]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"Participant {p + 1} has duplicate triple_ids")

    triple_counts: Dict[str, int] = {}
    for rows in assignments_by_participant:
        for r in rows:
            tid = r["triple_id"]
            triple_counts[tid] = triple_counts.get(tid, 0) + 1
    bad = [tid for tid, c in triple_counts.items() if c != 3]
    if bad:
        raise RuntimeError(f"Some triples were not assigned exactly 3 times. Example: {bad[:10]}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "participant_id",
                "item_order",
                "cycle",
                "triple_id",
                "relation_key",
                "A",
                "B",
                "C",
            ],
        )
        writer.writeheader()
        for p_idx, rows in enumerate(assignments_by_participant, start=1):
            rng.shuffle(rows)  # randomize within-participant display order
            for item_order, r in enumerate(rows, start=1):
                writer.writerow(
                    {
                        "participant_id": p_idx,
                        "item_order": item_order,
                        "cycle": r["cycle"],
                        "triple_id": r["triple_id"],
                        "relation_key": r["relation_key"],
                        "A": r["A"],
                        "B": r["B"],
                        "C": r["C"],
                    }
                )

    print(f"Wrote assignments: {args.out_csv}")
    print(f"Participants: {n_participants}, items each: {k}, triples: {n_triples}, repeats/triple: {repeats}")


if __name__ == "__main__":
    main()
