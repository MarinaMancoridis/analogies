"""
Identity task: pooled accuracy (is_identity) by prompt intervention.

Combines trial NDJSON sources (colon gold run + english/one_shot/few_shot variant run).
Plot order on the x-axis: English → Colon → One-shot → Few-shot.

Human reference: dashed line from human identity validation summary (mean correctness %),
or override with --human-accuracy.

  python interventions_identity.py
  python interventions_identity.py --variants-trials runs/<your_variant_run>/trials.ndjson
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
PROMPT_TYPES = ["colon", "english", "one_shot", "few_shot"]
# X-axis display order (data still keyed by prompt_type in NDJSON).
PLOT_ORDER = ["english", "colon", "one_shot", "few_shot"]
PLOT_LABELS = ["English", "Colon", "One-shot", "Few-shot"]

DEFAULT_COLON_TRIALS = HERE / "runs" / "gold_run" / "trials.ndjson"
DEFAULT_VARIANTS_TRIALS = HERE / "runs" / "20260504-134221-22411c" / "trials.ndjson"
DEFAULT_HUMAN_SUMMARY = HERE / "human_identity_responses" / "identity_human_agreement_summary.json"
DEFAULT_OUT = HERE / "figures" / "identity_interventions_prompt_accuracy.png"


def _accumulate_identity_by_prompt(paths: Sequence[Path]) -> Tuple[Dict[str, float], Dict[str, Tuple[int, int]]]:
    correct: Dict[str, int] = defaultdict(int)
    total: Dict[str, int] = defaultdict(int)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("error"):
                    continue
                pt = rec.get("prompt_type")
                if pt not in PROMPT_TYPES:
                    continue
                total[pt] += 1
                if rec.get("is_identity") is True:
                    correct[pt] += 1
    acc: Dict[str, float] = {}
    kn: Dict[str, Tuple[int, int]] = {}
    for pt in PROMPT_TYPES:
        n = total[pt]
        k = correct[pt]
        kn[pt] = (k, n)
        acc[pt] = (k / n) if n else 0.0
    return acc, kn


def _human_identity_fraction(summary_path: Path) -> Optional[float]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    overall = data.get("overall") or {}
    pct = overall.get("mean_correctness_pct")
    if pct is None:
        return None
    return float(pct) / 100.0


def _plot(
    kn: Dict[str, Tuple[int, int]],
    out_path: Path,
    human_y: Optional[float],
    human_legend: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = list(range(len(PLOT_ORDER)))
    ys = [(kn[pt][0] / kn[pt][1]) if kn[pt][1] else float("nan") for pt in PLOT_ORDER]
    labels = PLOT_LABELS

    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=150)
    ax.plot(
        xs,
        ys,
        marker="o",
        color="#ff7f0e",
        linewidth=2,
        markersize=8,
        label="Pooled model performance",
    )

    if human_y is not None:
        ax.axhline(
            human_y,
            linestyle="--",
            color="#2ca02c",
            linewidth=1.4,
            alpha=0.85,
            zorder=0,
            label=human_legend,
        )
    ax.legend(loc="lower right", framealpha=0.92, fontsize=9)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.7, 1.0)
    ax.set_xlabel("Prompt style")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.set_title("Prompting Interventions for Identity Task")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--colon-trials", type=Path, default=DEFAULT_COLON_TRIALS)
    ap.add_argument("--variants-trials", type=Path, default=DEFAULT_VARIANTS_TRIALS)
    ap.add_argument("--human-summary", type=Path, default=DEFAULT_HUMAN_SUMMARY)
    ap.add_argument("--human-accuracy", type=float, default=None, help="Override human line y.")
    ap.add_argument("--no-human-line", action="store_true")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    paths = [args.colon_trials, args.variants_trials]
    for p in paths:
        if not p.is_file():
            print(f"Missing: {p}", file=sys.stderr)
            return 1

    acc, kn = _accumulate_identity_by_prompt(paths)
    for pt in PROMPT_TYPES:
        k, n = kn[pt]
        if n == 0:
            print(f"Warning: no rows for prompt_type={pt}", file=sys.stderr)

    human_y: Optional[float] = None
    human_legend = ""
    if not args.no_human_line:
        if args.human_accuracy is not None:
            human_y = float(args.human_accuracy)
        elif args.human_summary.is_file():
            human_y = _human_identity_fraction(args.human_summary)
        if human_y is not None:
            human_legend = f"Human performance ({human_y:.1%})"

    _plot(kn, args.out, human_y, human_legend)
    print(f"Wrote {args.out}")
    for pt in PROMPT_TYPES:
        k, n = kn[pt]
        print(f"  {pt}: {acc[pt]:.4f} ({k}/{n})")
    if human_y is not None:
        print(f"  human line: {human_y:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
