"""
Test H0: lexical word-metrics do not improve prediction of completion accuracy,
after controlling for model and prompt type (and relation, for relational trials).

Each row is one completion from trial NDJSON. Lexical covariates are flattened
from A_metrics, B_metrics, C_metrics (identity: A and C only—B duplicates A;
relational: A–D). Numeric fields: len, pop_rank, pop_zipf, polysemy, brys
(missing brys imputed with column median for fitting).

Method:
- Pooled: nested binomial GLM (logit) MLE—likelihood ratio test (treats rows as i.i.d.).
- Clustered: same full model refit with cov_type='cluster'; joint Wald test that all
  lexical coefficients are zero (appropriate when many rows share the same analogy item).

Usage (from repo root):
  pip install -r hypothesis_testing/requirements.txt
  python hypothesis_testing/run_lexical_null_test.py

  python hypothesis_testing/run_lexical_null_test.py \\
    --identity-trials experiments/LLM_identity_completions/runs/gold_run/trials.ndjson \\
    --relational-trials experiments/LLM_relations_completions/runs/gold_curate_b/trials.ndjson
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import statsmodels.formula.api as smf
except ImportError as e:
    raise SystemExit(
        "statsmodels is required: pip install -r hypothesis_testing/requirements.txt"
    ) from e

REPO = Path(__file__).resolve().parents[1]

NUM_METRIC_KEYS = ("len", "pop_rank", "pop_zipf", "polysemy", "brys")


def _read_ndjson_rows(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return rows


def _flatten_metrics(prefix: str, m: Any) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not isinstance(m, dict):
        return out
    for k in NUM_METRIC_KEYS:
        col = f"{prefix}_{k}"
        v = m.get(k)
        if v is None:
            out[col] = np.nan
        else:
            try:
                out[col] = float(v)
            except (TypeError, ValueError):
                out[col] = np.nan
    return out


def _impute_median(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c]
        if s.notna().sum() == 0:
            df[c] = 0.0
        else:
            df[c] = s.fillna(s.median())


def _prune_lexical_columns(df: pd.DataFrame, lex: List[str]) -> List[str]:
    """Drop non-varying lexical columns (avoids singular design when e.g. all brys are missing)."""
    out: List[str] = []
    for c in lex:
        if c not in df.columns:
            continue
        if df[c].nunique(dropna=False) <= 1:
            continue
        out.append(c)
    # Drop pairwise identical columns (keep first).
    kept: List[str] = []
    for c in out:
        vc = df[c].to_numpy()
        if any(np.array_equal(vc, df[k].to_numpy()) for k in kept):
            continue
        kept.append(c)
    return kept


def build_identity_frame(rows: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """y = is_identity; lexical from A and C only (B equals A in gold identity)."""
    recs: List[Dict[str, Any]] = []
    for r in rows:
        if r.get("error"):
            continue
        if r.get("is_identity") is None:
            continue
        y = bool(r["is_identity"])
        tid = r.get("triple_id")
        row: Dict[str, Any] = {
            "y_acc": int(y),
            "model": str(r.get("model", "")),
            "prompt_type": str(r.get("prompt_type", "")),
            "triple_id": int(tid) if tid is not None else np.nan,
            "cluster_id": str(tid) if tid is not None else "",
        }
        if r.get("iteration") is not None:
            row["iteration"] = int(r["iteration"])
        row.update(_flatten_metrics("A", r.get("A_metrics")))
        row.update(_flatten_metrics("C", r.get("C_metrics")))
        recs.append(row)
    return pd.DataFrame.from_records(recs)


def build_relational_frame(rows: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """y = judge_majority_correct."""
    recs: List[Dict[str, Any]] = []
    for r in rows:
        if r.get("error"):
            continue
        j = r.get("judge_majority_correct")
        if j is None:
            continue
        rk = str(r.get("relation_key", ""))
        tid = r.get("triple_id")
        row: Dict[str, Any] = {
            "y_acc": int(bool(j)),
            "model": str(r.get("model", "")),
            "prompt_type": str(r.get("prompt_type", "")),
            "relation_key": rk,
            "triple_id": int(tid) if tid is not None else np.nan,
            "cluster_id": f"{rk}:{tid}" if tid is not None and rk else "",
        }
        for pref, key in (("A", "A_metrics"), ("B", "B_metrics"), ("C", "C_metrics"), ("D", "D_metrics")):
            row.update(_flatten_metrics(pref, r.get(key)))
        recs.append(row)
    return pd.DataFrame.from_records(recs)


def _lexical_cols(df: pd.DataFrame, prefixes: Sequence[str]) -> List[str]:
    cols: List[str] = []
    for p in prefixes:
        for k in NUM_METRIC_KEYS:
            c = f"{p}_{k}"
            if c in df.columns:
                cols.append(c)
    return cols


@dataclass
class LRResult:
    name: str
    n: int
    lr_stat: float
    df_diff: int
    p_value: float
    ll_null: float
    ll_full: float
    converged_null: bool
    converged_full: bool
    lexical_columns_used: List[str] = field(default_factory=list)
    cluster_key: str = ""
    n_clusters: int = 0
    wald_cluster_stat: float = float("nan")
    wald_cluster_df: int = 0
    wald_cluster_p_value: float = float("nan")


def _clustered_wald_lexical_zero(
    *,
    d: pd.DataFrame,
    full_formula: str,
    lex: List[str],
    groups: pd.Series,
) -> Tuple[float, int, float]:
    """
    Refit full logit with cluster-robust covariance; joint Wald chi-square that all
    lexical coefficients are zero. Returns (statistic, df, p-value).
    """
    from scipy.stats import chi2

    fit = smf.logit(full_formula, data=d).fit(
        disp=False,
        maxiter=250,
        cov_type="cluster",
        cov_kwds={"groups": groups},
    )
    params = fit.params
    cov = fit.cov_params()
    lex_set = set(lex)
    lex_params = [p for p in params.index if p in lex_set]
    if not lex_params:
        return float("nan"), 0, float("nan")
    idx = [params.index.get_loc(p) for p in lex_params]
    beta = params.values[np.asarray(idx, dtype=int)]
    v = cov.values[np.ix_(idx, idx)]
    cond = np.linalg.cond(v)
    if not np.isfinite(cond) or cond > 1e12:
        v_inv = np.linalg.pinv(v)
    else:
        v_inv = np.linalg.inv(v)
    stat = float(beta @ v_inv @ beta)
    df_w = len(lex_params)
    pval = float(chi2.sf(stat, df_w))
    return stat, df_w, pval


def _fit_lr_identity(df: pd.DataFrame) -> LRResult:
    lex = _lexical_cols(df, ("A", "C"))
    if not lex:
        raise ValueError("No lexical columns for identity frame")

    d = df.dropna(subset=["y_acc", "model", "prompt_type", "cluster_id"]).copy()
    d = d[d["cluster_id"].astype(str).str.len() > 0]
    _impute_median(d, lex)
    lex = _prune_lexical_columns(d, lex)
    if len(lex) < 1:
        raise ValueError("All lexical columns were constant after imputation; cannot test.")

    has_iter = "iteration" in d.columns and d["iteration"].nunique() > 1
    ctrl_formula = "y_acc ~ C(model) + C(prompt_type) + C(iteration)" if has_iter else "y_acc ~ C(model) + C(prompt_type)"
    lex_terms = " + ".join(lex)
    full_formula = ctrl_formula + " + " + lex_terms

    null = smf.logit(ctrl_formula, data=d).fit(disp=False, maxiter=200)
    full = smf.logit(full_formula, data=d).fit(disp=False, maxiter=200)
    lr = -2.0 * (null.llf - full.llf)
    df_diff = int(round(full.df_model - null.df_model))
    from scipy.stats import chi2

    p = float(chi2.sf(lr, df_diff)) if df_diff > 0 else float("nan")

    n_clust = int(d["cluster_id"].nunique())
    w_stat, w_df, w_p = _clustered_wald_lexical_zero(
        d=d, full_formula=full_formula, lex=lex, groups=d["cluster_id"]
    )

    return LRResult(
        name="identity",
        n=len(d),
        lr_stat=float(lr),
        df_diff=df_diff,
        p_value=p,
        ll_null=float(null.llf),
        ll_full=float(full.llf),
        converged_null=bool(null.mle_retvals.get("converged", True)),
        converged_full=bool(full.mle_retvals.get("converged", True)),
        lexical_columns_used=list(lex),
        cluster_key="triple_id (cluster = same identity analogy item across models/prompts/iterations)",
        n_clusters=n_clust,
        wald_cluster_stat=w_stat,
        wald_cluster_df=w_df,
        wald_cluster_p_value=w_p,
    )


def _fit_lr_relational(df: pd.DataFrame) -> LRResult:
    lex = _lexical_cols(df, ("A", "B", "C", "D"))
    if not lex:
        raise ValueError("No lexical columns for relational frame")

    d = df.dropna(subset=["y_acc", "model", "prompt_type", "relation_key", "cluster_id"]).copy()
    d = d[d["cluster_id"].astype(str).str.len() > 0]
    _impute_median(d, lex)
    lex = _prune_lexical_columns(d, lex)
    if len(lex) < 1:
        raise ValueError("All lexical columns were constant after imputation; cannot test.")

    ctrl_formula = "y_acc ~ C(model) + C(prompt_type) + C(relation_key)"
    lex_terms = " + ".join(lex)
    full_formula = ctrl_formula + " + " + lex_terms

    null = smf.logit(ctrl_formula, data=d).fit(disp=False, maxiter=200)
    full = smf.logit(full_formula, data=d).fit(disp=False, maxiter=200)
    lr = -2.0 * (null.llf - full.llf)
    df_diff = int(round(full.df_model - null.df_model))
    from scipy.stats import chi2

    p = float(chi2.sf(lr, df_diff)) if df_diff > 0 else float("nan")

    n_clust = int(d["cluster_id"].nunique())
    w_stat, w_df, w_p = _clustered_wald_lexical_zero(
        d=d, full_formula=full_formula, lex=lex, groups=d["cluster_id"]
    )

    return LRResult(
        name="relational",
        n=len(d),
        lr_stat=float(lr),
        df_diff=df_diff,
        p_value=p,
        ll_null=float(null.llf),
        ll_full=float(full.llf),
        converged_null=bool(null.mle_retvals.get("converged", True)),
        converged_full=bool(full.mle_retvals.get("converged", True)),
        lexical_columns_used=list(lex),
        cluster_key="relation_key:triple_id (cluster = same relational analogy item)",
        n_clusters=n_clust,
        wald_cluster_stat=w_stat,
        wald_cluster_df=w_df,
        wald_cluster_p_value=w_p,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--identity-trials",
        type=Path,
        action="append",
        default=[],
        help="Identity trials NDJSON (repeatable). Default: gold_run only.",
    )
    ap.add_argument(
        "--relational-trials",
        type=Path,
        default=REPO / "experiments/LLM_relations_completions/runs/gold_curate_b/trials.ndjson",
        help="Relational trials NDJSON.",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=REPO / "hypothesis_testing" / "lexical_null_lr_results.json",
        help="Write numeric summary JSON here.",
    )
    args = ap.parse_args()

    id_paths: List[Path] = list(args.identity_trials)
    if not id_paths:
        id_paths = [REPO / "experiments/LLM_identity_completions/runs/gold_run/trials.ndjson"]

    print("Identity trial files:")
    for p in id_paths:
        print(f"  {p}")
    print("Relational trial file:")
    print(f"  {args.relational_trials}")

    summaries: List[Dict[str, Any]] = []

    id_rows = _read_ndjson_rows(id_paths)
    id_df = build_identity_frame(id_rows)
    if id_df.empty:
        print("No identity rows; skip identity test.", file=sys.stderr)
    else:
        print(f"\nIdentity completions (usable rows before fit): {len(id_df)}")
        try:
            res_id = _fit_lr_identity(id_df)
            print(
                f"Pooled LR (i.i.d. rows): H0 lexical block useless given model/prompt[/iter] — "
                f"LR={res_id.lr_stat:.4f}, df={res_id.df_diff}, p={res_id.p_value:.4g}, n={res_id.n}"
            )
            print(f"  log-lik null={res_id.ll_null:.3f} full={res_id.ll_full:.3f} "
                  f"converged_null={res_id.converged_null} converged_full={res_id.converged_full}")
            print(
                f"Cluster-robust Wald (K={res_id.n_clusters} clusters): {res_id.cluster_key}\n"
                f"  H0: all lexical coeffs = 0 | W={res_id.wald_cluster_stat:.4f}, "
                f"df={res_id.wald_cluster_df}, p={res_id.wald_cluster_p_value:.4g}"
            )
            summaries.append(asdict(res_id))
        except Exception as e:
            print(f"Identity LR test failed: {e}", file=sys.stderr)
            summaries.append({"name": "identity", "error": str(e)})

    if not args.relational_trials.is_file():
        print(f"Missing relational file {args.relational_trials}; skip relational test.", file=sys.stderr)
    else:
        rel_rows = _read_ndjson_rows([args.relational_trials])
        rel_df = build_relational_frame(rel_rows)
        print(f"\nRelational completions (usable rows before fit): {len(rel_df)}")
        try:
            res_rel = _fit_lr_relational(rel_df)
            print(
                f"Pooled LR (i.i.d. rows): H0 lexical block useless given model/prompt/relation — "
                f"LR={res_rel.lr_stat:.4f}, df={res_rel.df_diff}, p={res_rel.p_value:.4g}, n={res_rel.n}"
            )
            print(
                f"  log-lik null={res_rel.ll_null:.3f} full={res_rel.ll_full:.3f} "
                f"converged_null={res_rel.converged_null} converged_full={res_rel.converged_full}"
            )
            print(
                f"Cluster-robust Wald (K={res_rel.n_clusters} clusters): {res_rel.cluster_key}\n"
                f"  H0: all lexical coeffs = 0 | W={res_rel.wald_cluster_stat:.4f}, "
                f"df={res_rel.wald_cluster_df}, p={res_rel.wald_cluster_p_value:.4g}"
            )
            summaries.append(asdict(res_rel))
        except Exception as e:
            print(f"Relational LR test failed: {e}", file=sys.stderr)
            summaries.append({"name": "relational", "error": str(e)})

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    out_payload = {
        "h0_description": (
            "Lexical numeric metrics (len, pop_rank, pop_zipf, polysemy, brys per word role) "
            "have all coefficients jointly zero, given control factors. "
            "Pooled LR treats rows as i.i.d. Clustered Wald refits the full logit with "
            "cov_type='cluster' and tests the same joint hypothesis with cluster-robust covariance "
            "(identity: cluster=triple_id; relational: cluster=relation_key:triple_id)."
        ),
        "identity_trial_paths": [str(p.resolve()) for p in id_paths],
        "relational_trial_path": str(args.relational_trials.resolve()) if args.relational_trials.is_file() else None,
        "results": summaries,
    }
    args.out_json.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
