"""
Plot pooled model accuracy by prompt intervention (colon → english → one-shot → few-shot).

**Relational** trials (default: gold_curate_b): micro-average of judge_majority_correct
(relation-following), all models pooled.

**Identity** trials (optional, default: LLM_identity_completions gold_run): micro-average of
is_identity (model repeats C as the answer). The current gold identity run uses **colon
prompts only** (see run_identity_triples.PROMPT_TYPE); other interventions appear as gaps
unless you add more runs.

Human baselines (optional):
- Relational: LLM panel majority on human completions (--human-panel), or --human-accuracy.
- Identity: mean correctness from human validation summary JSON (--human-identity-summary),
  or --human-identity-accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
PROMPT_TYPES = ["colon", "english", "one_shot", "few_shot"]

DEFAULT_RELATIONAL_TRIALS = HERE / "runs" / "gold_curate_b" / "trials.ndjson"
DEFAULT_IDENTITY_TRIALS = (
    HERE.parent / "LLM_identity_completions" / "runs" / "gold_run" / "trials.ndjson"
)
DEFAULT_HUMAN_REL_PANEL = (
    HERE.parent / "validate_relation_following" / "human_judgments_panel.ndjson"
)
DEFAULT_HUMAN_IDENTITY_SUMMARY = (
    HERE.parent
    / "LLM_identity_completions"
    / "human_identity_responses"
    / "identity_human_agreement_summary.json"
)
DEFAULT_OUT = HERE / "figures" / "interventions_prompt_accuracy.png"


def pooled_relational_by_prompt(
    trials_path: Path,
) -> Tuple[Dict[str, float], Dict[str, Tuple[int, int]]]:
    """Micro-average judge_majority_correct per prompt_type."""
    correct: Dict[str, int] = defaultdict(int)
    total: Dict[str, int] = defaultdict(int)
    with trials_path.open("r", encoding="utf-8") as f:
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
            if rec.get("judge_majority_correct") is True:
                correct[pt] += 1
    acc: Dict[str, float] = {}
    kn: Dict[str, Tuple[int, int]] = {}
    for pt in PROMPT_TYPES:
        n = total[pt]
        k = correct[pt]
        kn[pt] = (k, n)
        acc[pt] = (k / n) if n else 0.0
    return acc, kn


def pooled_identity_by_prompt(
    trials_path: Path,
) -> Tuple[Dict[str, float], Dict[str, Tuple[int, int]]]:
    """Micro-average is_identity per prompt_type."""
    correct: Dict[str, int] = defaultdict(int)
    total: Dict[str, int] = defaultdict(int)
    with trials_path.open("r", encoding="utf-8") as f:
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


def human_panel_majority_rate(panel_path: Path) -> Optional[float]:
    """≥2/3 panel judges mark judge_majority_correct (relational human completions)."""
    votes: Dict[str, List[bool]] = defaultdict(list)
    with panel_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("record_type") != "human_completion_judgment":
                continue
            cid = rec.get("completion_id")
            if not isinstance(cid, str):
                continue
            votes[cid].append(rec.get("judge_majority_correct") is True)
    ok = tot = 0
    for ys in votes.values():
        if len(ys) != 3:
            continue
        tot += 1
        if sum(ys) >= 2:
            ok += 1
    if tot == 0:
        return None
    return ok / tot


def human_identity_mean_correctness(summary_path: Path) -> Optional[float]:
    """Mean correctness % from identity human validation summary → fraction in [0, 1]."""
    with summary_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    overall = data.get("overall") or {}
    pct = overall.get("mean_correctness_pct")
    if pct is None:
        return None
    return float(pct) / 100.0


def plot_prompt_curves(
    rel_acc: Dict[str, float],
    id_acc: Optional[Dict[str, float]],
    id_kn: Optional[Dict[str, Tuple[int, int]]],
    out_path: Path,
    human_rel_y: Optional[float],
    human_rel_legend: str,
    human_id_y: Optional[float],
    human_id_legend: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = list(range(len(PROMPT_TYPES)))
    labels = ["Colon", "English", "One-shot", "Few-shot"]
    ys_rel = [rel_acc[pt] for pt in PROMPT_TYPES]

    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=150)
    ax.plot(
        xs,
        ys_rel,
        marker="o",
        color="#1f77b4",
        linewidth=2,
        markersize=8,
        label="Relational (relation-following)",
    )

    if id_acc is not None and id_kn is not None:
        ys_id = []
        for pt in PROMPT_TYPES:
            k, n = id_kn[pt]
            ys_id.append((k / n) if n else float("nan"))
        ax.plot(
            xs,
            ys_id,
            marker="^",
            color="#ff7f0e",
            linewidth=2,
            markersize=8,
            label="Identity (repeat C; colon-only in current data)",
        )

    handles_done = False

    if human_rel_y is not None:
        ax.axhline(
            human_rel_y,
            linestyle="--",
            color="0.45",
            linewidth=1.2,
            alpha=0.45,
            zorder=0,
            label=human_rel_legend,
        )
        handles_done = True
    if human_id_y is not None:
        ax.axhline(
            human_id_y,
            linestyle=(0, (3, 4)),
            color="0.35",
            linewidth=1.2,
            alpha=0.55,
            zorder=0,
            label=human_id_legend,
        )
        handles_done = True

    if handles_done:
        ax.legend(loc="lower right", framealpha=0.92, fontsize=8)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (pooled across models)")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Prompt intervention")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.set_title("Prompt interventions: relational rules vs identity analogies")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--trials",
        type=Path,
        default=DEFAULT_RELATIONAL_TRIALS,
        help="Relational trial NDJSON (default: gold_curate_b).",
    )
    ap.add_argument(
        "--identity-trials",
        type=Path,
        default=DEFAULT_IDENTITY_TRIALS,
        help="Identity trial NDJSON (default: LLM_identity_completions gold_run).",
    )
    ap.add_argument(
        "--no-identity",
        action="store_true",
        help="Do not plot identity series or identity human baseline.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output PNG path.",
    )
    ap.add_argument(
        "--human-panel",
        type=Path,
        default=DEFAULT_HUMAN_REL_PANEL,
        help="Relational human panel NDJSON (ignored if --human-accuracy is set).",
    )
    ap.add_argument(
        "--human-accuracy",
        type=float,
        default=None,
        help="Fixed y for relational human reference line (overrides panel).",
    )
    ap.add_argument(
        "--no-human-line",
        action="store_true",
        help="No relational human reference line.",
    )
    ap.add_argument(
        "--human-identity-summary",
        type=Path,
        default=DEFAULT_HUMAN_IDENTITY_SUMMARY,
        help="identity_human_agreement_summary.json for human identity benchmark.",
    )
    ap.add_argument(
        "--human-identity-accuracy",
        type=float,
        default=None,
        help="Fixed y for identity human reference line (overrides summary JSON).",
    )
    ap.add_argument(
        "--no-human-identity-line",
        action="store_true",
        help="No identity human reference line.",
    )
    args = ap.parse_args()

    if not args.trials.is_file():
        print(f"Missing relational trials file: {args.trials}", file=sys.stderr)
        return 1

    rel_acc, rel_kn = pooled_relational_by_prompt(args.trials)
    id_acc: Optional[Dict[str, float]] = None
    id_kn: Optional[Dict[str, Tuple[int, int]]] = None
    if not args.no_identity:
        if args.identity_trials.is_file():
            id_acc, id_kn = pooled_identity_by_prompt(args.identity_trials)
            nonempty = sum(1 for pt in PROMPT_TYPES if id_kn[pt][1] > 0)
            print(
                f"Identity trials: {nonempty} prompt type(s) with data "
                f"(from {args.identity_trials})",
                file=sys.stderr,
            )
        else:
            print(
                f"No identity trials at {args.identity_trials} (--no-identity to silence).",
                file=sys.stderr,
            )

    human_rel_y: Optional[float] = None
    human_rel_legend = ""
    if not args.no_human_line:
        if args.human_accuracy is not None:
            human_rel_y = float(args.human_accuracy)
            human_rel_legend = f"Human relational (fixed): {human_rel_y:.3f}"
        elif args.human_panel.is_file():
            human_rel_y = human_panel_majority_rate(args.human_panel)
            if human_rel_y is not None:
                human_rel_legend = f"Human relational (panel majority): {human_rel_y:.3f}"
        if human_rel_y is None and args.human_accuracy is None and not args.no_human_line:
            print(
                f"No relational human line (missing panel: {args.human_panel}).",
                file=sys.stderr,
            )

    human_id_y: Optional[float] = None
    human_id_legend = ""
    if not args.no_identity and not args.no_human_identity_line and id_kn is not None:
        if args.human_identity_accuracy is not None:
            human_id_y = float(args.human_identity_accuracy)
            human_id_legend = f"Human identity (fixed): {human_id_y:.3f}"
        elif args.human_identity_summary.is_file():
            human_id_y = human_identity_mean_correctness(args.human_identity_summary)
            if human_id_y is not None:
                human_id_legend = f"Human identity (validation summary): {human_id_y:.3f}"
        if human_id_y is None and args.human_identity_accuracy is None:
            print(
                f"No identity human line (missing summary: {args.human_identity_summary}).",
                file=sys.stderr,
            )

    plot_prompt_curves(
        rel_acc,
        id_acc,
        id_kn,
        args.out,
        human_rel_y,
        human_rel_legend,
        human_id_y,
        human_id_legend,
    )
    print(f"Wrote {args.out}")
    print("Relational (judge majority):")
    for pt in PROMPT_TYPES:
        k, n = rel_kn[pt]
        print(f"  {pt}: {rel_acc[pt]:.4f} ({k}/{n})")
    if id_kn is not None:
        print("Identity (is_identity):")
        for pt in PROMPT_TYPES:
            k, n = id_kn[pt]
            if n:
                print(f"  {pt}: {id_acc[pt]:.4f} ({k}/{n})")
            else:
                print(f"  {pt}: (no trials)")
    if human_rel_y is not None:
        print(f"Human relational reference: {human_rel_y:.4f}")
    if human_id_y is not None:
        print(f"Human identity reference: {human_id_y:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
