from __future__ import annotations

import argparse
import json
from pathlib import Path


def _default_run_dir(runs_root: Path) -> Path:
    gold = runs_root / "gold_run"
    if gold.exists():
        return gold
    manifests = sorted(runs_root.glob("*/run_manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not manifests:
        raise FileNotFoundError(f"No runs found under {runs_root}")
    return manifests[0].parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print incorrect GPT identity completions from trials.ndjson."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="Path to a specific run dir. Defaults to runs/gold_run if present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent
    runs_root = here / "runs"
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else _default_run_dir(runs_root)

    trials_path = run_dir / "trials.ndjson"
    if not trials_path.exists():
        raise FileNotFoundError(f"Missing trials file: {trials_path}")

    total_false = 0
    print(f"Run: {run_dir}")
    print("Incorrect GPT completions:\n")

    with trials_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            model = str(row.get("model", ""))
            if "gpt" not in model.lower():
                continue
            if row.get("is_identity"):
                continue

            total_false += 1
            print(
                f"- model={model} triple_id={row.get('triple_id')} iter={row.get('iteration')} "
                f"A={row.get('A')} C={row.get('C')} expected={row.get('expected')} "
                f"parsed={row.get('parsed_answer')} error={row.get('error')}"
            )
            raw = (row.get("raw_response") or "").replace("\n", "\\n")
            print(f"  raw_response={raw}\n")

    print(f"Total incorrect GPT completions: {total_false}")


if __name__ == "__main__":
    main()
