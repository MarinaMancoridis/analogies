# Exploratory embedding figures — captions

## Sample counts (authoritative)

- **Human responses embedded:** **1300** (`participant_eligible_complete`, non-empty `D`, non-empty after `clean_answer`).
- **LLM responses embedded:** **4983** (`gold_curate_b` trials, `prompt_type == colon`, no `error`, non-empty `D` after `clean_answer`).
- **Embedding model:** `text-embedding-3-small`
- **Participation ratio (global PCA):** 41.28

## Figures

### `fig01_pca_human_vs_llm.png`

PCA (first two components) of standardized OpenAI embeddings of **cleaned** completion tokens. Blue: human Prolific/Qualtrics completions (eligible only); orange: LLM completions from gold_curate_b **colon** trials. Overlap is expected when both solve the same analogy; separation would suggest systematic geometric bias. Exploratory only.

### `fig02_pca_scree_cumulative.png`

Cumulative variance explained by PCA components. High intrinsic dimensionality is typical for sentence/lexical embeddings; use with PC scatter for qualitative structure only.

### `fig03_pc1_density_by_source.png`

Kernel-free histogram of PC1 scores. Discrete Shannon entropy of binned PC1 (32 bins): human≈3.225, LLM≈3.265. Higher entropy suggests a more spread-out marginal along the leading direction; interpret cautiously (binning, N).

### `fig04_embedding_norm_log.png`

Distribution of embedding L2 norms (log1p transform). Systematic shifts can indicate length/frequency effects or model-specific scaling; geometry comparisons should primarily use standardized PCA space.

### `fig05_cleaned_token_char_length.png`

Character length of **clean_answer** outputs. Humans and parsers can yield multi-character strings; mismatch in length distributions confounds raw embedding comparisons—PCA on standardized features partly mitigates this.

### `fig06_pca_facets_by_relation.png`

Same global PCA as Fig.1, faceted by **relation_key**. Shows whether human–LLM geometry differs consistently within relation types. Cell titles truncated.

### `fig07_centroid_l2_pca10.png`

L2 distance between mean PCA vectors (first 10 components). Magnitude 2.2101 is exploratory; statistical testing would require paired design or hierarchical models.

### `fig08_pairwise_cosine_subsample.png`

Subsampled pairwise cosine similarities in **raw** embedding space. Tighter within-LLM cloud would echo “homogeneity” narratives (e.g. Hivemind-style analyses); human cloud often wider if responses are noisier or more diverse.

### `fig09_participation_ratio_by_source.png`

Participation ratio Tr(λ)²/Tr(λ²) from PCA eigenvalues on each subset (standardized embeddings). Higher suggests a more “spread” covariance spectrum; compare only qualitatively (different N).

### `fig10_tsne_subsample.png`

t-SNE on a stratified subsample (cap ~1200, max_iter=500) for nonlinear 2-D visualization. Non-metric; use for pattern discovery only, not distances or densities.

### `fig11_pc1_boxplot_by_relation.png`

PC1 distributions per **relation_key**: for each relation, two boxplots (blue = human, orange = LLM). Shows relation-specific shifts along the main global axis.
