# Analogies — reproducing the main experiments

This repository holds code and **frozen artifacts** (trial logs, human exports, gold run directories) for the identity completion task, relational completion with relation-following validation, and human–LLM preference evaluation. You can **recompute tables from committed data** without calling any APIs, or **regenerate LLM trials** when the required API keys are set.

## Environment

- **Python:** 3.10+ recommended (3.11 used in project venv).
- **Install dependencies** (minimal set for the experiment scripts):

  ```bash
  pip install openai anthropic google-generativeai google-api-core together tqdm
  ```

- **Import path:** modules are imported as `analogies.*`, so the **parent directory of this repo folder** must be on `PYTHONPATH`. If this repository is checked out as `analogies/`, use:

  ```bash
  export ANALOGIES_REPO="$(pwd)"   # after: cd /path/to/analogies
  export PYTHONPATH="$(dirname "$ANALOGIES_REPO"):$PYTHONPATH"
  ```

  All `python ...` commands below assume you have run these two lines from inside the repo (or set the variables equivalently).

## API keys (only if you generate new LLM outputs)

Inference is routed by model in `constants.py` (`models_to_developer`). Keys are read in `constants.py` from the environment:

| Variable | Used for |
|----------|-----------|
| `OPENAI_API_KEY` | Native OpenAI models (e.g. `gpt-4o`) |
| `OPENROUTER_API_KEY` | Most chat models in the paper (Llama, GPT-4.1, Gemini, Claude, DeepSeek, GPT-5.4-mini, …) |
| `TOGETHER_API_KEY` | Together-hosted models |
| `GEMINI_API_KEY` | Legacy Gemini path in `utils.py` (most Gemini calls here use OpenRouter) |
| `CLAUDE_API_KEY` | Legacy Anthropic path (most Claude calls use OpenRouter) |

Example (bash):

```bash
export OPENAI_API_KEY="sk-..."
export OPENROUTER_API_KEY="sk-or-..."
export TOGETHER_API_KEY="..."   # if you use Together models
# export GEMINI_API_KEY="..."  # rarely needed for current model list
# export CLAUDE_API_KEY="..."  # rarely needed for current model list
```

Without these, you can still run **summarization and LaTeX table scripts** against the committed `runs/` data.

**Cost / scale (full regenerations):**

- **Relational gold run** (`gold_curate_b`): 500 triples × 10 models × 4 prompt types = **20,000** generations, each with relation self-judging (`DEFAULT_JUDGE_N = 1` in `run_relation_triples.py`).
- **Identity gold run**: 100 triples × 10 models × 3 iterations = **3,000** generations.
- **Preference LLM judges**: default **100** pairwise judgments per evaluator model (×10 models); restrict with flags as needed.

---

## Main result tables (committed numbers)

Sources: `experiments/LLM_identity_completions/runs/gold_run/latex_table_main_results.tex`, `experiments/validate_relation_following/relation_following_validation_table.tex`, `experiments/preference_LLM_humans/llm_preference_table_colon.tex` (colon prompt filter + human survey row).

### Identity copying (accuracy)

| Model | Accuracy |
|-------|----------|
| Claude-Sonnet-4.5 | 0.93 (0.01) |
| Claude-Opus-4.6 | 0.65 (0.03) |
| Gemini-3-Flash | 0.88 (0.02) |
| Gemini-3.1-Flash-Lite | 0.89 (0.02) |
| GPT-4o | 0.92 (0.02) |
| GPT-4.1 | 0.88 (0.02) |
| GPT-5.4-Mini | 0.83 (0.02) |
| DeepSeek-V3 | 0.76 (0.02) |
| DeepSeek-V3.2 | 0.82 (0.02) |
| Llama-3.3 | 0.36 (0.03) |

*Binomial-style standard errors in parentheses; see script for exact definitions.*

### Relation-following (automated judge majority correct, all prompt types; `gold_curate_b`)

Proportions with SE in parentheses. Columns: **Attr.** attribute, **Case** case relations, **C–P** cause–purpose, **C–Inc.** class–inclusion, **Contr.** contrast, **Non-att.** non-attribute, **P–W** part–whole, **Ref.** reference, **Sim.** similar, **S–T** space–time, **Mean** pooled over relations.

| Model | Attr. | Case | C–P | C–Inc. | Contr. | Non-att. | P–W | Ref. | Sim. | S–T | Mean |
|-------|-------|------|-----|--------|--------|----------|-----|------|------|-----|------|
| Llama 3.3 70B Instruct | 0.74 (0.031) | 0.81 (0.028) | 0.57 (0.035) | 0.69 (0.033) | 0.95 (0.015) | 0.89 (0.023) | 0.68 (0.033) | 0.98 (0.009) | 0.93 (0.018) | 0.84 (0.026) | **0.81 (0.009)** |
| GPT-4o | 0.75 (0.031) | 0.69 (0.033) | 0.41 (0.035) | 0.24 (0.030) | 0.98 (0.010) | 0.86 (0.024) | 0.28 (0.032) | 0.57 (0.035) | 0.77 (0.030) | 0.70 (0.032) | **0.63 (0.011)** |
| GPT-4.1 | 0.70 (0.032) | 0.59 (0.035) | 0.34 (0.033) | 0.27 (0.031) | 0.89 (0.022) | 0.81 (0.028) | 0.32 (0.033) | 0.82 (0.027) | 0.75 (0.031) | 0.67 (0.033) | **0.62 (0.011)** |
| GPT-5.4 Mini | 0.52 (0.035) | 0.41 (0.035) | 0.33 (0.033) | 0.48 (0.035) | 0.94 (0.017) | 0.47 (0.035) | 0.44 (0.035) | 0.90 (0.021) | 0.92 (0.020) | 0.48 (0.035) | **0.59 (0.011)** |
| Gemini 3 Flash Preview | 0.69 (0.033) | 0.74 (0.031) | 0.28 (0.032) | 0.54 (0.035) | 0.97 (0.011) | 0.78 (0.029) | 0.27 (0.031) | 0.80 (0.028) | 0.84 (0.026) | 0.81 (0.028) | **0.67 (0.010)** |
| Gemini 3.1 Flash-Lite Preview | 0.73 (0.031) | 0.76 (0.030) | 0.41 (0.035) | 0.43 (0.035) | 0.93 (0.019) | 0.57 (0.035) | 0.43 (0.035) | 0.71 (0.032) | 0.83 (0.027) | 0.59 (0.035) | **0.64 (0.011)** |
| Claude Sonnet 4.5 | 0.45 (0.035) | 0.47 (0.035) | 0.14 (0.025) | 0.41 (0.035) | 0.83 (0.026) | 0.53 (0.035) | 0.23 (0.030) | 0.54 (0.035) | 0.83 (0.027) | 0.51 (0.035) | **0.49 (0.011)** |
| Claude Opus 4.6 | 0.51 (0.035) | 0.45 (0.035) | 0.33 (0.033) | 0.55 (0.035) | 0.88 (0.023) | 0.74 (0.031) | 0.41 (0.035) | 0.75 (0.031) | 0.91 (0.020) | 0.63 (0.034) | **0.61 (0.011)** |
| DeepSeek V3 | 0.22 (0.029) | 0.36 (0.034) | 0.15 (0.025) | 0.08 (0.019) | 0.81 (0.028) | 0.17 (0.027) | 0.09 (0.020) | 0.15 (0.025) | 0.84 (0.026) | 0.33 (0.033) | **0.32 (0.010)** |
| DeepSeek V3.2 | 0.24 (0.030) | 0.33 (0.033) | 0.15 (0.026) | 0.10 (0.021) | 0.75 (0.031) | 0.23 (0.030) | 0.10 (0.021) | 0.30 (0.033) | 0.74 (0.031) | 0.26 (0.031) | **0.32 (0.010)** |
| **Human** | **0.18 (0.034)** | **0.36 (0.041)** | **0.17 (0.033)** | **0.15 (0.031)** | **0.61 (0.043)** | **0.16 (0.032)** | **0.22 (0.036)** | **0.42 (0.043)** | **0.48 (0.044)** | **0.27 (0.039)** | **0.30 (0.013)** |

*Human row: eligible human completions; each cell is the fraction with ≥2/3 “Correct” among three LLM judges (`gpt-5.4-mini`, `gemini-3.1-flash-lite-preview`, `deepseek-v3.2`).*

### Pairwise preference (fraction choosing human completion; colon LLM completions only)

| Evaluator | Human-chosen proportion |
|-----------|-------------------------|
| Llama 3.3 70B Instruct | 0.19 (0.039) |
| GPT-4o | 0.22 (0.041) |
| GPT-4.1 | 0.17 (0.038) |
| GPT-5.4 Mini | 0.26 (0.044) |
| Gemini 3 Flash Preview | 0.16 (0.037) |
| Gemini 3.1 Flash-Lite Preview | 0.21 (0.041) |
| Claude Sonnet 4.5 | 0.2 (0.04) |
| Claude Opus 4.6 | 0.23 (0.042) |
| DeepSeek V3 | 0.3 (0.046) |
| DeepSeek V3.2 | 0.2 (0.04) |
| **Human** (survey aggregate) | **0.41 (0.018)** |

*An unfiltered LLM-judge-only table is in `experiments/preference_LLM_humans/llm_preference_table.tex`.*

---

## Commands: recompute tables from existing trial files (no API calls)

From the repo root, with `PYTHONPATH` set as above:

**Relation-following summary and LaTeX table**

```bash
python experiments/validate_relation_following/LLM_validate_relation_following.py \
  --trials-ndjson experiments/LLM_relations_completions/runs/gold_curate_b/trials.ndjson

python experiments/validate_relation_following/latex_validation_table.py \
  --in-json experiments/validate_relation_following/llm_validate_relation_following.json \
  --human-panel experiments/validate_relation_following/human_judgments_panel.ndjson \
  --out-tex experiments/validate_relation_following/relation_following_validation_table.tex
```

**Identity LaTeX tables**

```bash
python experiments/LLM_identity_completions/make_latex_tables.py \
  --run-dir experiments/LLM_identity_completions/runs/gold_run
```

**Preference: human survey analysis + LLM judge table (colon filter)**

```bash
python experiments/preference_LLM_humans/analyze_human_preference_responses.py \
  --responses-csv experiments/preference_LLM_humans/human_analogy_comparisons.csv \
  --out-json experiments/preference_LLM_humans/human_preference_analysis.json

python experiments/preference_LLM_humans/make_llm_preference_table.py \
  --results-ndjson experiments/preference_LLM_humans/pairwise_llm_judgments_colon.ndjson \
  --human-analysis-json experiments/preference_LLM_humans/human_preference_analysis.json \
  --out-tex experiments/preference_LLM_humans/llm_preference_table_colon.tex
```

---

## Commands: regenerate static triples (deterministic; no API)

Default seeds match the committed JSON (`12345`; relational: 50 triples per relation).

```bash
python static_triples/relational/generate_relation_triples.py
python static_triples/identity/generate_identity_triples.py
```

---

## Commands: full LLM runs (requires API keys)

Use `--serial` to run all models in one process (default `full` mode otherwise spawns separate terminal windows on macOS). Outputs go to a **new timestamped directory** under `experiments/.../runs/`; replace or symlink to `gold_run` / `gold_curate_b` if you want paths to match the scripts’ defaults.

**Relational completions (matches `gold_curate_b` manifest: seed 12345, fixed B from Qualtrics merge CSV)**

```bash
python experiments/LLM_relations_completions/run_relation_triples.py --mode full --serial --seed 12345 \
  --fixed-b-csv experiments/LLM_relations_completions/qualtrics_loop_and_merge/relation_loop_and_merge_all.csv
```

**Identity completions (matches `gold_run`: 100 triples × 10 models × 3 iters, seed 12345)**

```bash
python experiments/LLM_identity_completions/run_identity_triples.py --mode full --serial --seed 12345
```

**Human completion judging (for the “Human” row of the relation table)** — build the flat file first, then run three judge shards (API keys required):

```bash
python experiments/validate_relation_following/validate_human_responses.py --only-eligible

./experiments/validate_relation_following/run_human_judgment_panel.sh smoke
# or: ./experiments/validate_relation_following/run_human_judgment_panel.sh full
```

**Preference: rebuild gold links and rerun LLM pairwise judges**

```bash
python experiments/preference_LLM_humans/build_gold_standard_links.py

python experiments/preference_LLM_humans/run_pairwise_preferences.py --seed 12345 \
  --required-llm-prompt-type colon
```

Then rerun `analyze_human_preference_responses.py` and `make_llm_preference_table.py` as in the no-API section.

---

## Layout reference

| Path | Role |
|------|------|
| `experiments/LLM_relations_completions/runs/gold_curate_b/` | Frozen relational trial NDJSON + manifest |
| `experiments/LLM_identity_completions/runs/gold_run/` | Frozen identity trial NDJSON + manifest |
| `experiments/validate_relation_following/` | Relation-following aggregation + human judge panel |
| `experiments/preference_LLM_humans/` | Preference construction, LLM judges, human CSV analysis |
| `static_triples/` | Generators and committed `identity_triples.json`, `relation_triples.json` |
