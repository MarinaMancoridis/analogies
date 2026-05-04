#!/usr/bin/env python3
"""
Exploratory embedding geometry for human vs LLM analogy completions (colon prompt).

Uses OpenAI text embeddings (default: text-embedding-3-large). Requires:
  pip install openai numpy matplotlib scikit-learn
  export OPENAI_API_KEY=...

Mock run (no API, random vectors for pipeline test):
  python run_embedding_exploration.py --mock

Re-use saved embeddings (no API; figures + captions only):
  python run_embedding_exploration.py --from-npz entropy_analysis/artifacts/embeddings_bundle.npz

Apples-to-apples preprocessing: both corpora pass through the same ``clean_answer``
normalization used elsewhere in the repo (extract primary token / ANSWER: pattern).

Outputs under ./figures/ and ./artifacts/; see FIGURE_CAPTIONS.md for interpretive captions.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def clean_answer(resp: str) -> str:
    """Same logic as ``common.clean_answer`` (ANSWER: pattern else first word token)."""
    resp = (resp or "").strip()
    m = re.search(r"ANSWER:\s*([A-Za-z\-']+)", resp, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    m2 = re.search(r"\b([A-Za-z\-']+)\b", resp)
    return m2.group(1).lower() if m2 else ""


@dataclass
class Row:
    text_raw: str
    text_clean: str
    source: str  # "human" | "llm"
    relation_key: str
    triple_id: int
    meta: Dict[str, Any]


def _load_human_rows(flat_path: Path) -> List[Row]:
    data = json.loads(flat_path.read_text(encoding="utf-8"))
    rows: List[Row] = []
    for c in data["completions"]:
        if not c.get("participant_eligible_complete"):
            continue
        raw = str(c.get("D") or "").strip()
        if not raw:
            continue
        cl = clean_answer(raw)
        if not cl:
            continue
        rows.append(
            Row(
                text_raw=raw,
                text_clean=cl,
                source="human",
                relation_key=str(c["relation_key"]),
                triple_id=int(c["triple_id"]),
                meta={"completion_id": c.get("completion_id")},
            )
        )
    return rows


def _load_llm_rows(trials_path: Path) -> List[Row]:
    rows: List[Row] = []
    with trials_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("prompt_type") != "colon":
                continue
            if r.get("error"):
                continue
            raw = str(r.get("D") or "").strip()
            if not raw:
                continue
            cl = clean_answer(raw)
            if not cl:
                continue
            rows.append(
                Row(
                    text_raw=raw,
                    text_clean=cl,
                    source="llm",
                    relation_key=str(r["relation_key"]),
                    triple_id=int(r["triple_id"]),
                    meta={"model": r.get("model")},
                )
            )
    return rows


def _embed_openai(texts: List[str], model: str, batch_size: int) -> np.ndarray:
    from openai import OpenAI

    n = len(texts)
    n_batches = (n + batch_size - 1) // batch_size
    print(
        f"Embedding {n} texts via OpenAI in {n_batches} request(s) "
        f"(model={model!r}, batch_size={batch_size}). This can take several minutes; "
        "there is no output between batches until now.",
        flush=True,
    )
    client = OpenAI()
    vecs: List[List[float]] = []
    for b, i in enumerate(range(0, n, batch_size)):
        batch = texts[i : i + batch_size]
        t0 = time.perf_counter()
        print(f"  embeddings request {b + 1}/{n_batches} ({len(batch)} strings)...", flush=True)
        resp = client.embeddings.create(model=model, input=batch)
        for d in resp.data:
            vecs.append(d.embedding)
        dt = time.perf_counter() - t0
        print(f"    done in {dt:.1f}s", flush=True)
        time.sleep(0.02)
    print("All embedding requests finished.", flush=True)
    return np.array(vecs, dtype=np.float64)


def _embed_mock(texts: List[str], dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Correlated structure by text hash so PCA is non-degenerate
    base = rng.standard_normal((len(texts), dim))
    for i, t in enumerate(texts):
        h = hash(t) % (2**32)
        skew = (h % 17) / 100.0
        base[i] += skew
    return base


def _shannon_entropy_1d(x: np.ndarray, n_bins: int = 32) -> float:
    hist, _ = np.histogram(x, bins=n_bins, density=True)
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 1e-12]
    return float(-(p * np.log(p)).sum())


def _participation_ratio(evals: np.ndarray) -> float:
    """Effective dimensionality from eigenvalues of covariance."""
    ev = np.maximum(evals, 0)
    s = ev.sum()
    if s <= 0:
        return 0.0
    return float((s**2) / (np.sum(ev**2) + 1e-12))


def _paired_triple_human_llm_distances(
    Xs: np.ndarray,
    triple_ids: np.ndarray,
    sources: np.ndarray,
    relations: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[int, float],
    Dict[int, str],
]:
    """
    For each human completion, within the same triple_id as LLM colon completions:
    - d_centroid: ℓ₂ distance in **standardized** embedding space to the mean of LLM vectors.
    - d_min: ℓ₂ distance to the **nearest** LLM vector in that triple.

    Returns per-human arrays (d_centroid, d_min, relation, triple_id) and per-triple
    mean(d_centroid) with relation labels for aggregation plots.
    """
    d_cent_list: List[float] = []
    d_min_list: List[float] = []
    rel_h: List[str] = []
    tid_h: List[int] = []
    triple_to_dcent_vals: Dict[int, List[float]] = {}

    for tid in np.unique(triple_ids):
        mh = (triple_ids == tid) & (sources == "human")
        ml = (triple_ids == tid) & (sources == "llm")
        if not np.any(mh) or not np.any(ml):
            continue
        idx_h = np.where(mh)[0]
        idx_l = np.where(ml)[0]
        L = Xs[idx_l]
        centroid = L.mean(axis=0)
        rk = str(relations[idx_h[0]])
        vals_t: List[float] = []
        for hi in idx_h:
            xh = Xs[hi]
            dc = float(np.linalg.norm(xh - centroid))
            dm = float(np.linalg.norm(L - xh, axis=1).min())
            d_cent_list.append(dc)
            d_min_list.append(dm)
            rel_h.append(rk)
            tid_h.append(int(tid))
            vals_t.append(dc)
        triple_to_dcent_vals[int(tid)] = vals_t

    trip_mean_centroid = {t: float(np.mean(v)) for t, v in triple_to_dcent_vals.items()}
    trip_relation = {
        t: str(relations[np.where((triple_ids == t) & (sources == "human"))[0][0]])
        for t in trip_mean_centroid
    }
    return (
        np.array(d_cent_list, dtype=np.float64),
        np.array(d_min_list, dtype=np.float64),
        np.array(rel_h),
        np.array(tid_h, dtype=np.int64),
        trip_mean_centroid,
        trip_relation,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Random embeddings; no API.")
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-large",
        help="OpenAI embedding model id.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--human-json",
        type=Path,
        default=REPO
        / "experiments"
        / "validate_relation_following"
        / "human_completions_flat.json",
    )
    parser.add_argument(
        "--llm-trials",
        type=Path,
        default=REPO
        / "experiments"
        / "LLM_relations_completions"
        / "runs"
        / "gold_curate_b"
        / "trials.ndjson",
    )
    parser.add_argument(
        "--from-npz",
        type=Path,
        default=None,
        help=(
            "Load embeddings from this .npz (e.g. artifacts/embeddings_bundle.npz) "
            "and skip the API. Incompatible with --mock."
        ),
    )
    args = parser.parse_args()
    if args.mock and args.from_npz:
        raise SystemExit("Use either --mock or --from-npz, not both.")

    fig_dir = HERE / "figures"
    art_dir = HERE / "artifacts"
    fig_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    bundle_model: str
    loaded_from_npz = args.from_npz is not None

    if loaded_from_npz:
        if not args.from_npz.is_file():
            raise SystemExit(f"--from-npz file not found: {args.from_npz}")
        print(f"Loading embeddings from {args.from_npz} (no API calls).", flush=True)
        data = np.load(args.from_npz, allow_pickle=True)
        X = np.asarray(data["X"], dtype=np.float64)
        sources = np.asarray(data["sources"])
        relations = np.asarray(data["relations"])
        triple_ids = np.asarray(data["triple_ids"])
        texts = np.asarray(data["texts"], dtype=object)
        if "embedding_model" in data.files:
            bundle_model = str(np.asarray(data["embedding_model"]).item())
        else:
            bundle_model = args.embedding_model
        n_h = int(np.sum(sources == "human"))
        n_l = int(np.sum(sources == "llm"))
        meta_counts = {
            "n_human_eligible_nonempty_cleaned": n_h,
            "n_llm_colon_no_error_nonempty_cleaned": n_l,
            "human_source": f"from_npz:{args.from_npz}",
            "llm_source": f"from_npz:{args.from_npz}",
            "preprocessing": "clean_answer (logic matches common.clean_answer; same for both corpora)",
            "llm_filter": "(see npz bundle)",
            "human_filter": "(see npz bundle)",
            "unique_triple_id_human": int(len(np.unique(triple_ids[sources == "human"]))),
            "unique_triple_id_llm": int(len(np.unique(triple_ids[sources == "llm"]))),
        }
    else:
        human_rows = _load_human_rows(args.human_json)
        llm_rows = _load_llm_rows(args.llm_trials)

        texts = [r.text_clean for r in human_rows + llm_rows]
        sources = np.array([r.source for r in human_rows + llm_rows])
        relations = np.array([r.relation_key for r in human_rows + llm_rows])
        triple_ids = np.array([r.triple_id for r in human_rows + llm_rows])

        n_h = int((sources == "human").sum())
        n_l = int((sources == "llm").sum())

        meta_counts = {
            "n_human_eligible_nonempty_cleaned": n_h,
            "n_llm_colon_no_error_nonempty_cleaned": n_l,
            "human_source": str(args.human_json),
            "llm_source": str(args.llm_trials),
            "preprocessing": "clean_answer (logic matches common.clean_answer; same for both corpora)",
            "llm_filter": "prompt_type == colon, error is null/empty, D nonempty after clean",
            "human_filter": "participant_eligible_complete, D nonempty after clean",
            "unique_triple_id_human": len({r.triple_id for r in human_rows}),
            "unique_triple_id_llm": len({r.triple_id for r in llm_rows}),
        }

    (art_dir / "sample_counts.json").write_text(
        json.dumps(meta_counts, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta_counts, indent=2), flush=True)

    if loaded_from_npz:
        pass  # X already set
    elif args.mock:
        dim = 256
        X = _embed_mock(list(texts), dim=dim, seed=args.seed)
        bundle_model = f"mock_dim{dim}"
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("Set OPENAI_API_KEY or use --mock.")
        X = _embed_openai(list(texts), args.embedding_model, args.batch_size)
        bundle_model = args.embedding_model

    if not loaded_from_npz:
        print("Saving embeddings bundle and computing PCA / figures...", flush=True)
        np.savez_compressed(
            art_dir / "embeddings_bundle.npz",
            X=X,
            sources=sources,
            relations=relations,
            triple_ids=triple_ids,
            texts=np.asarray(texts, dtype=object),
            embedding_model=bundle_model,
        )
    else:
        print("Computing PCA / figures (bundle not re-written)...", flush=True)

    # --- PCA (standardize then PCA, common in LM representation papers) ---
    from matplotlib import pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    plt.rcParams.update({"font.size": 10, "figure.figsize": (7, 5.5)})

    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(50, X.shape[1], X.shape[0] - 1), random_state=args.seed)
    Z = pca.fit_transform(Xs)
    evr = pca.explained_variance_ratio_
    pr_all = _participation_ratio(pca.explained_variance_)

    captions: List[Tuple[str, str]] = []

    # Fig 1: PCA human vs LLM
    fig, ax = plt.subplots()
    for src, color, label in [
        ("human", "#1f77b4", f"Human (n={n_h})"),
        ("llm", "#ff7f0e", f"LLM colon (n={n_l})"),
    ]:
        m = sources == src
        ax.scatter(
            Z[m, 0],
            Z[m, 1],
            s=10,
            alpha=0.35,
            c=color,
            label=label,
            edgecolors="none",
        )
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}% var.)")
    ax.legend(markerscale=2)
    ax.set_title("PCA of completion embeddings (standardized)")
    p1 = fig_dir / "fig01_pca_human_vs_llm.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=160)
    plt.close(fig)
    captions.append(
        (
            str(p1.name),
            "PCA (first two components) of standardized OpenAI embeddings of **cleaned** "
            "completion tokens. Blue: human Prolific/Qualtrics completions (eligible only); "
            "orange: LLM completions from gold_curate_b **colon** trials. Overlap is expected "
            "when both solve the same analogy; separation would suggest systematic geometric "
            "bias. Exploratory only.",
        )
    )

    # Fig 2: scree
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(evr) + 1), np.cumsum(evr) * 100, "o-", ms=3)
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative variance explained (%)")
    ax.set_title("PCA cumulative variance")
    ax.axhline(90, color="gray", ls="--", lw=0.8)
    p2 = fig_dir / "fig02_pca_scree_cumulative.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=160)
    plt.close(fig)
    captions.append(
        (
            str(p2.name),
            "Cumulative variance explained by PCA components. High intrinsic dimensionality "
            "is typical for sentence/lexical embeddings; use with PC scatter for qualitative "
            "structure only.",
        )
    )

    # Fig 3: PC1 distribution by source
    fig, ax = plt.subplots()
    for src, color in [("human", "#1f77b4"), ("llm", "#ff7f0e")]:
        m = sources == src
        ax.hist(Z[m, 0], bins=40, alpha=0.45, color=color, label=src, density=True)
    ax.set_xlabel("PC1")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title("PC1 distribution: human vs LLM")
    p3 = fig_dir / "fig03_pc1_density_by_source.png"
    fig.tight_layout()
    fig.savefig(p3, dpi=160)
    plt.close(fig)
    h_pc1 = _shannon_entropy_1d(Z[sources == "human", 0])
    l_pc1 = _shannon_entropy_1d(Z[sources == "llm", 0])
    captions.append(
        (
            str(p3.name),
            f"Kernel-free histogram of PC1 scores. Discrete Shannon entropy of binned PC1 "
            f"(32 bins): human≈{h_pc1:.3f}, LLM≈{l_pc1:.3f}. Higher entropy suggests a more "
            f"spread-out marginal along the leading direction; interpret cautiously (binning, N).",
        )
    )

    # Fig 4: embedding L2 norm (original space)
    norms = np.linalg.norm(X, axis=1)
    fig, ax = plt.subplots()
    for src, color in [("human", "#1f77b4"), ("llm", "#ff7f0e")]:
        m = sources == src
        ax.hist(np.log1p(norms[m]), bins=35, alpha=0.45, color=color, label=src, density=True)
    ax.set_xlabel("log(1 + ||e||₂) raw embedding")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title("Embedding norm (log scale)")
    p4 = fig_dir / "fig04_embedding_norm_log.png"
    fig.tight_layout()
    fig.savefig(p4, dpi=160)
    plt.close(fig)
    captions.append(
        (
            str(p4.name),
            "Distribution of embedding L2 norms (log1p transform). Systematic shifts can "
            "indicate length/frequency effects or model-specific scaling; geometry comparisons "
            "should primarily use standardized PCA space.",
        )
    )

    # Fig 5: token length (cleaned)
    lens = np.array([len(t) for t in texts])
    fig, ax = plt.subplots()
    for src, color in [("human", "#1f77b4"), ("llm", "#ff7f0e")]:
        m = sources == src
        ax.hist(lens[m], bins=np.arange(0.5, max(lens) + 2), alpha=0.45, color=color, label=src, density=True)
    ax.set_xlabel("Character length of cleaned token")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title("Cleaned completion length")
    p5 = fig_dir / "fig05_cleaned_token_char_length.png"
    fig.tight_layout()
    fig.savefig(p5, dpi=160)
    plt.close(fig)
    captions.append(
        (
            str(p5.name),
            "Character length of **clean_answer** outputs. Humans and parsers can yield "
            "multi-character strings; mismatch in length distributions confounds raw "
            "embedding comparisons—PCA on standardized features partly mitigates this.",
        )
    )

    # Fig 6: mean PC vector per relation, human vs llm (first 2D)
    rel_keys = sorted(set(relations.tolist()))
    fig, axes = plt.subplots(2, 5, figsize=(14, 6), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, rk in zip(axes, rel_keys):
        mh = (sources == "human") & (relations == rk)
        ml = (sources == "llm") & (relations == rk)
        ax.scatter(Z[mh, 0], Z[mh, 1], s=8, c="#1f77b4", alpha=0.35, label="H")
        ax.scatter(Z[ml, 0], Z[ml, 1], s=8, c="#ff7f0e", alpha=0.25, label="L")
        ax.set_title(rk[:12] + ("…" if len(rk) > 12 else ""), fontsize=8)
        ax.axhline(0, color="k", lw=0.2, alpha=0.3)
        ax.axvline(0, color="k", lw=0.2, alpha=0.3)
    axes[0].legend(markerscale=2, fontsize=7)
    fig.suptitle("PCA by relation_key (shared PC basis)")
    p6 = fig_dir / "fig06_pca_facets_by_relation.png"
    fig.tight_layout()
    fig.savefig(p6, dpi=160)
    plt.close(fig)
    captions.append(
        (
            str(p6.name),
            "Same global PCA as Fig.1, faceted by **relation_key**. Shows whether human–LLM "
            "geometry differs consistently within relation types. Cell titles truncated.",
        )
    )

    # Fig 7: centroid distance — mean embedding per source in PCA space
    ch = Z[sources == "human"].mean(axis=0)
    cl = Z[sources == "llm"].mean(axis=0)
    dist = float(np.linalg.norm(ch[:10] - cl[:10]))
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["|mean PC_human − mean PC_LLM|₂\n(first 10 dims)"], [dist], color="#2ca02c")
    ax.set_ylabel("Distance")
    ax.set_title("Global centroid separation (PCA space)")
    p7 = fig_dir / "fig07_centroid_l2_pca10.png"
    fig.tight_layout()
    fig.savefig(p7, dpi=160)
    plt.close(fig)
    captions.append(
        (
            str(p7.name),
            f"L2 distance between mean PCA vectors (first 10 components). "
            f"Magnitude {dist:.4f} is exploratory; statistical testing would require "
            f"paired design or hierarchical models.",
        )
    )

    # Fig 8: within-source pairwise cosine (subsample)
    rng = np.random.default_rng(args.seed)
    def sample_cosine(idx: np.ndarray, k: int = 800) -> np.ndarray:
        idx = np.where(idx)[0]
        if len(idx) < 2:
            return np.array([])
        cos: List[float] = []
        for _ in range(k):
            i, j = rng.choice(idx, 2, replace=False)
            a, b = X[i], X[j]
            cos.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
        return np.array(cos)

    ch_cos = sample_cosine(sources == "human")
    cl_cos = sample_cosine(sources == "llm")
    cross: List[float] = []
    ih = np.where(sources == "human")[0]
    il_ = np.where(sources == "llm")[0]
    for _ in range(800):
        i = rng.choice(ih)
        j = rng.choice(il_)
        a, b = X[i], X[j]
        cross.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
    cross_arr = np.array(cross)

    fig, ax = plt.subplots()
    ax.hist(ch_cos, bins=30, alpha=0.45, density=True, label="within human", color="#1f77b4")
    ax.hist(cl_cos, bins=30, alpha=0.45, density=True, label="within LLM", color="#ff7f0e")
    ax.hist(cross_arr, bins=30, alpha=0.35, density=True, label="human–LLM pairs", color="#9467bd")
    ax.set_xlabel("Cosine similarity (raw embedding)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title("Random pairwise cosine similarities (subsampled)")
    p8 = fig_dir / "fig08_pairwise_cosine_subsample.png"
    fig.tight_layout()
    fig.savefig(p8, dpi=160)
    plt.close(fig)
    captions.append(
        (
            str(p8.name),
            "Subsampled pairwise cosine similarities in **raw** embedding space. Tighter "
            "within-LLM cloud would echo “homogeneity” narratives (e.g. Hivemind-style analyses); "
            "human cloud often wider if responses are noisier or more diverse.",
        )
    )

    # Fig 9: participation ratio by source
    def pr_subset(mask: np.ndarray) -> float:
        if mask.sum() < 5:
            return float("nan")
        sub = Xs[mask]
        pca_s = PCA(n_components=min(30, sub.shape[1], sub.shape[0] - 1))
        pca_s.fit(sub)
        return _participation_ratio(pca_s.explained_variance_)

    pr_h = pr_subset(sources == "human")
    pr_l = pr_subset(sources == "llm")
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.bar(["human", "llm"], [pr_h, pr_l], color=["#1f77b4", "#ff7f0e"])
    ax.set_ylabel("Participation ratio (PCA evals)")
    ax.set_title("Effective dimensionality (subset PCA)")
    p9 = fig_dir / "fig09_participation_ratio_by_source.png"
    fig.tight_layout()
    fig.savefig(p9, dpi=160)
    plt.close(fig)
    captions.append(
        (
            str(p9.name),
            "Participation ratio Tr(λ)²/Tr(λ²) from PCA eigenvalues on each subset "
            "(standardized embeddings). Higher suggests a more “spread” covariance spectrum; "
            "compare only qualitatively (different N).",
        )
    )

    # Fig 10: t-SNE on stratified subsample (cap keeps runtime reasonable on CPU)
    max_n = 1200
    if len(texts) > max_n:
        idx_h = np.where(sources == "human")[0]
        idx_l = np.where(sources == "llm")[0]
        take_h = rng.choice(idx_h, size=min(len(idx_h), max_n // 2), replace=False)
        take_l = rng.choice(idx_l, size=min(len(idx_l), max_n // 2), replace=False)
        sub_idx = np.concatenate([take_h, take_l])
    else:
        sub_idx = np.arange(len(texts))
    Xsub = Xs[sub_idx]
    sub_src = sources[sub_idx]
    perplexity = min(30, max(5, (len(sub_idx) - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=args.seed,
        max_iter=500,
    )
    T = tsne.fit_transform(Xsub)
    fig, ax = plt.subplots()
    for src, color in [("human", "#1f77b4"), ("llm", "#ff7f0e")]:
        m = sub_src == src
        ax.scatter(T[m, 0], T[m, 1], s=12, alpha=0.4, c=color, label=src, edgecolors="none")
    ax.set_title(f"t-SNE subsample (n={len(sub_idx)})")
    ax.legend()
    ax.set_xticks([])
    ax.set_yticks([])
    p10 = fig_dir / "fig10_tsne_subsample.png"
    fig.tight_layout()
    fig.savefig(p10, dpi=160)
    plt.close(fig)
    captions.append(
        (
            str(p10.name),
            "t-SNE on a stratified subsample (cap ~1200, max_iter=500) for nonlinear 2-D visualization. "
            "Non-metric; use for pattern discovery only, not distances or densities.",
        )
    )

    # Fig 11: box PC1 by relation for human vs llm (paired at relation level visually)
    fig, ax = plt.subplots(figsize=(12, 4))
    step = 3.0
    for i, rk in enumerate(rel_keys):
        mh = Z[(sources == "human") & (relations == rk), 0]
        ml = Z[(sources == "llm") & (relations == rk), 0]
        b1 = ax.boxplot(
            mh,
            positions=[i * step],
            widths=0.7,
            showfliers=False,
            patch_artist=True,
        )
        for b in b1["boxes"]:
            b.set_facecolor("#1f77b4")
            b.set_alpha(0.55)
        b2 = ax.boxplot(
            ml,
            positions=[i * step + 1.0],
            widths=0.7,
            showfliers=False,
            patch_artist=True,
        )
        for b in b2["boxes"]:
            b.set_facecolor("#ff7f0e")
            b.set_alpha(0.55)
    ax.set_xticks([i * step + 0.5 for i in range(len(rel_keys))])
    ax.set_xticklabels(rel_keys, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("PC1")
    ax.set_title("PC1 by relation (blue = human, orange = LLM per relation)")
    p11 = fig_dir / "fig11_pc1_boxplot_by_relation.png"
    fig.tight_layout()
    fig.savefig(p11, dpi=160)
    plt.close(fig)
    captions.append(
        (
            str(p11.name),
            "PC1 distributions per **relation_key**: for each relation, two boxplots (blue = human, "
            "orange = LLM). Shows relation-specific shifts along the main global axis.",
        )
    )

    # --- Paired-by-triple: each human vs LLM completions for the same triple_id (standardized Xs)
    d_cent, d_min, _, _, trip_mean_c, trip_rel_c = _paired_triple_human_llm_distances(
        Xs, triple_ids, sources, relations
    )
    paired_summ = {
        "n_human_completions_paired": int(len(d_cent)),
        "n_triples_with_both_sources": int(len(trip_mean_c)),
        "median_dist_human_to_llm_centroid": float(np.median(d_cent)),
        "median_dist_human_to_nearest_llm": float(np.median(d_min)),
        "mean_dist_human_to_llm_centroid": float(np.mean(d_cent)),
        "mean_dist_human_to_nearest_llm": float(np.mean(d_min)),
    }
    (art_dir / "paired_triple_summary.json").write_text(
        json.dumps(paired_summ, indent=2), encoding="utf-8"
    )

    # Fig 12: histograms of paired distances (easy read)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax0 = axes[0]
    ax0.hist(d_cent, bins=48, color="#1f77b4", alpha=0.78, edgecolor="white", linewidth=0.4)
    ax0.axvline(
        float(np.median(d_cent)),
        color="crimson",
        ls="--",
        lw=2,
        label=f"median = {float(np.median(d_cent)):.3f}",
    )
    ax0.set_xlabel(r"$\ell_2$ distance to mean of LLM vectors (same triple)")
    ax0.set_ylabel("Human completions (count)")
    ax0.set_title("Distance to LLM cloud center")
    ax0.legend(loc="upper right", fontsize=9)

    ax1 = axes[1]
    ax1.hist(d_min, bins=48, color="#2ca02c", alpha=0.78, edgecolor="white", linewidth=0.4)
    ax1.axvline(
        float(np.median(d_min)),
        color="crimson",
        ls="--",
        lw=2,
        label=f"median = {float(np.median(d_min)):.3f}",
    )
    ax1.set_xlabel(r"$\ell_2$ distance to nearest LLM vector (same triple)")
    ax1.set_ylabel("Human completions (count)")
    ax1.set_title("Distance to closest LLM draw")
    ax1.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        "Paired-by-triple (standardized embeddings): each human vs only LLM completions\n"
        "for the same analogy item — smaller distances ⇒ human sits closer to the LLM cluster.",
        fontsize=11,
        y=1.03,
    )
    fig.tight_layout()
    p12 = fig_dir / "fig12_paired_dist_hist_to_llm_cloud.png"
    fig.savefig(p12, dpi=160, bbox_inches="tight")
    plt.close(fig)
    captions.append(
        (
            str(p12.name),
            "Two histograms over **all eligible human completions**: (left) ℓ₂ distance in **standardized** "
            "embedding space from the human vector to the **mean of all LLM vectors in the same triple_id**; "
            "(right) distance to the **nearest** LLM vector in that triple. "
            "This matches the experimental design (same item) better than pooling unrelated analogies.",
        )
    )

    # Fig 13: one value per triple = mean (over humans in that triple) distance to LLM centroid
    rel_keys_paired = sorted(set(trip_rel_c.values()))
    data_by_rel: List[List[float]] = []
    labels_rel: List[str] = []
    for rk in rel_keys_paired:
        vals = [trip_mean_c[t] for t in trip_mean_c if trip_rel_c[t] == rk]
        if vals:
            data_by_rel.append(vals)
            labels_rel.append(f"{rk}\n(n={len(vals)} triples)")
    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    bp = ax.boxplot(data_by_rel, patch_artist=True, showfliers=False)
    for box in bp["boxes"]:
        box.set_facecolor("#aec7e8")
        box.set_alpha(0.85)
    ax.set_xticklabels(labels_rel, rotation=38, ha="right", fontsize=8)
    ax.set_ylabel("Per-triple mean: human → LLM-centroid distance\n(standardized ℓ₂)")
    ax.set_title("Which relations show humans farther from the LLM cloud center? (one box per relation)")
    fig.tight_layout()
    p13 = fig_dir / "fig13_paired_mean_dist_by_relation.png"
    fig.savefig(p13, dpi=160, bbox_inches="tight")
    plt.close(fig)
    captions.append(
        (
            str(p13.name),
            "Each point is one **triple_id**: the **mean** (over human completions for that triple) of "
            "distance-to-LLM-centroid. Boxplots group triples by **relation_key**. "
            "Higher boxes ⇒ humans tend to sit farther from where LLM completions concentrate for that relation.",
        )
    )

    # Fig 14: 3×3 example triples — local PCA (this triple only) to show clustering
    valid_tids = [
        int(t)
        for t in np.unique(triple_ids)
        if int(np.sum((triple_ids == t) & (sources == "human"))) >= 1
        and int(np.sum((triple_ids == t) & (sources == "llm"))) >= 4
    ]
    if len(valid_tids) <= 9:
        chosen_triples = np.array(valid_tids, dtype=np.int64)
    else:
        chosen_triples = rng.choice(np.array(valid_tids, dtype=np.int64), size=9, replace=False)
    fig, axes = plt.subplots(3, 3, figsize=(10.2, 10.2))
    legend_ax = None
    for k in range(9):
        ax = axes.ravel()[k]
        if k >= len(chosen_triples):
            ax.axis("off")
            continue
        tid = int(chosen_triples[k])
        mh = (triple_ids == tid) & (sources == "human")
        ml = (triple_ids == tid) & (sources == "llm")
        idx_h = np.where(mh)[0]
        idx_l = np.where(ml)[0]
        rk = str(relations[idx_h[0]])
        rows = [Xs[i] for i in idx_h] + [Xs[i] for i in idx_l]
        sub = np.vstack(rows)
        if sub.shape[0] < 3:
            ax.axis("off")
            continue
        pca_t = PCA(n_components=2, random_state=args.seed + k)
        P = pca_t.fit_transform(sub)
        n_hi = len(idx_h)
        Ph, Pl = P[:n_hi], P[n_hi:]
        cent = Pl.mean(axis=0)
        use_lbl = legend_ax is None
        ax.scatter(
            Pl[:, 0],
            Pl[:, 1],
            c="#ff7f0e",
            s=36,
            alpha=0.55,
            label="LLM" if use_lbl else "",
            edgecolors="none",
        )
        ax.scatter(
            Ph[:, 0],
            Ph[:, 1],
            c="#1f77b4",
            s=140,
            marker="*",
            zorder=5,
            label="Human" if use_lbl else "",
            edgecolors="black",
            linewidths=0.35,
        )
        ax.scatter(
            [cent[0]],
            [cent[1]],
            c="crimson",
            marker="X",
            s=110,
            zorder=6,
            linewidths=1.2,
            label="LLM mean" if use_lbl else "",
        )
        if use_lbl:
            legend_ax = ax
        short_rk = rk[:22] + "…" if len(rk) > 22 else rk
        ax.set_title(f"triple {tid}\n{short_rk}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Nine example triples — PCA fit **only** to points from that triple\n"
        "★ human · orange = LLM draws · ✕ = mean LLM — tight orange cloud = similar LLM answers",
        fontsize=11,
        y=1.01,
    )
    if legend_ax is not None:
        handles, leg_labels = legend_ax.get_legend_handles_labels()
        fig.legend(handles, leg_labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.99), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p14 = fig_dir / "fig14_paired_example_triples_local_pca.png"
    fig.savefig(p14, dpi=160, bbox_inches="tight")
    plt.close(fig)
    captions.append(
        (
            str(p14.name),
            "**Local** 2D PCA per triple (not the global Fig.1 basis): orange dots are LLM colon completions, "
            "stars are human completions, crimson ✕ is their mean. "
            "When the orange cloud is tight and the star sits inside or near it, human and LLM geometry align "
            "for that item; a distant star is an intuitive “mismatch” visualization.",
        )
    )

    cap_path = HERE / "FIGURE_CAPTIONS.md"
    lines = [
        "# Exploratory embedding figures — captions",
        "",
        "## Sample counts (authoritative)",
        "",
        f"- **Human responses embedded:** **{n_h}** "
        "(`participant_eligible_complete`, non-empty `D`, non-empty after `clean_answer`).",
        f"- **LLM responses embedded:** **{n_l}** "
        "(`gold_curate_b` trials, `prompt_type == colon`, no `error`, non-empty `D` after `clean_answer`).",
        f"- **Embedding model:** `{bundle_model}`" + (" (MOCK)" if args.mock else ""),
        f"- **Participation ratio (global PCA):** {pr_all:.2f}",
        "",
        "## Figures",
        "",
    ]
    for fname, cap in captions:
        lines.append(f"### `{fname}`")
        lines.append("")
        lines.append(cap)
        lines.append("")
    cap_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {cap_path}")


if __name__ == "__main__":
    main()
