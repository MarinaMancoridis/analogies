import os, json, random, importlib, datetime
from analogies.common import make_run_dir, append_jsonl, write_summary, load_brysbaert_norms
BRYS_PATH = "./analogies/concreteness.txt" 

MODEL = "gpt-5"
N_TRIALS = 1
RNG = random.Random()
N_TRIALS_PER_TYPE = 10

# ---- Config toggles ----
MODE = "per_type"     # "per_type" or "weighted" (mixture)
POS_LIST = ["noun", "verb", "adjective", "adverb"]

# ---- Spec builders ----
def build_identity_specs(
    *,
    include_pop_rare=True,
    include_same_pos=True,
    include_cross_pos=True,
    include_cooccurring=True,       
    include_noncooccurring=True,    
    include_concreteness=True,
    weight=1
) -> list[dict]:
    specs: list[dict] = []
    if include_pop_rare:
        specs += [
            {"key": "identity_pop_pop",   "module": "analogies.analogy_types.identity",
             "kwargs": {"A_mode": "popular", "C_mode": "popular"}, "weight": weight},
            {"key": "identity_pop_rare",  "module": "analogies.analogy_types.identity",
             "kwargs": {"A_mode": "popular", "C_mode": "rare"},    "weight": weight},
            {"key": "identity_rare_pop",  "module": "analogies.analogy_types.identity",
             "kwargs": {"A_mode": "rare",    "C_mode": "popular"}, "weight": weight},
            {"key": "identity_rare_rare", "module": "analogies.analogy_types.identity",
             "kwargs": {"A_mode": "rare",    "C_mode": "rare"},    "weight": weight},
        ]
    if include_same_pos:
        for pos in POS_LIST:
            specs.append({
                "key": f"identity_pos_{pos}",
                "module": "analogies.analogy_types.identity",
                "kwargs": {"A_mode": f"pos:{pos}", "C_mode": f"pos:{pos}"},
                "weight": weight,
            })
    if include_cross_pos:
        for a in POS_LIST:
            for c in POS_LIST:
                if a == c:
                    continue
                specs.append({
                    "key": f"identity_pos_{a}_to_{c}",
                    "module": "analogies.analogy_types.identity",
                    "kwargs": {"A_mode": f"pos:{a}", "C_mode": f"pos:{c}"},
                    "weight": weight,
                })
    if include_cooccurring:
        specs.append({
            "key": "identity_cooccurring",
            "module": "analogies.analogy_types.identity",
            "kwargs": {"A_mode": "cooccurring", "C_mode": "cooccurring"},
            "weight": weight,
        })
    if include_noncooccurring:
        specs.append({
            "key": "identity_noncooccurring",
            "module": "analogies.analogy_types.identity",
            "kwargs": {"A_mode": "noncooccurring", "C_mode": "noncooccurring"},
            "weight": weight,
        })
    if include_concreteness:
        specs += [
            {"key": "identity_abs_abs", "module": "analogies.analogy_types.identity",
             "kwargs": {"A_mode": "ac:abstract", "C_mode": "ac:abstract"}, "weight": weight},
            {"key": "identity_abs_conc", "module": "analogies.analogy_types.identity",
             "kwargs": {"A_mode": "ac:abstract", "C_mode": "ac:concrete"}, "weight": weight},
            {"key": "identity_conc_abs", "module": "analogies.analogy_types.identity",
             "kwargs": {"A_mode": "ac:concrete", "C_mode": "ac:abstract"}, "weight": weight},
            {"key": "identity_conc_conc", "module": "analogies.analogy_types.identity",
             "kwargs": {"A_mode": "ac:concrete", "C_mode": "ac:concrete"}, "weight": weight},
        ]
    return specs

def build_cyclic_specs(*, weight=1) -> list[dict]:
    return [{
        "key": "cyclic",
        "module": "analogies.analogy_types.cyclic",
        "kwargs": {},
        "weight": weight,
    }]

def build_pair_specs(*, weight=1) -> list[dict]:
    return [{
        "key": "pair",
        "module": "analogies.analogy_types.pair_analogy",
        "kwargs": {},
        "weight": weight,
    }]

def build_specs(
    *,
    use_identity=True,
    use_cyclic=True,
    use_pair=True,
    identity_opts=None,
    weights=None,
) -> list[dict]:
    """
    identity_opts: dict for identity flags (include_pop_rare, include_same_pos, include_cross_pos)
    weights: optional dict like {"identity": 2, "cyclic": 1, "pair": 1}
    """
    identity_opts = identity_opts or {
        "include_pop_rare": True,
        "include_same_pos": True,
        "include_cross_pos": True,
    }
    weights = weights or {}
    out: list[dict] = []
    if use_identity:
        out += build_identity_specs(weight=weights.get("identity", 1), **identity_opts)
    if use_cyclic:
        out += build_cyclic_specs(weight=weights.get("cyclic", 1))
    if use_pair:
        out += build_pair_specs(weight=weights.get("pair", 1))
    return out

# ------------- (for run-time) build_specs -------------
# Build the full menu (identity + cyclic + pair) -> where you can toggle the use of each type
ANALOGY_SPECS = build_specs(
    use_identity=True,
    use_cyclic=False,
    use_pair=False,
    identity_opts={
        "include_pop_rare": False,
        "include_same_pos": False,
        "include_cross_pos": False,  # ← A≠C variants
        "include_cooccurring": False,     
        "include_noncooccurring": False,  
        "include_concreteness": True,
    },
    weights={"identity": 2, "cyclic": 1, "pair": 1},  # used only in weighted mode
)


# ---- Loaders: per-type (deterministic) vs mixture (weighted) ----
def load_runners_per_type(specs):
    """Deterministic per-type execution. Keeps kwargs."""
    runners = {}
    for s in specs:
        mod = importlib.import_module(s["module"])
        run = getattr(mod, "run_trial")
        runners[s["key"]] = {
            "run": run,
            "weight": s.get("weight", 1),
            "kwargs": s.get("kwargs", {}),
        }
    return runners

def load_runners_weighted(specs):
    """
    Mixture loader: returns (runners, chooser) where chooser() gives a key
    sampled by weights across ALL entries (identity variants + cyclic + pair).
    """
    runners = load_runners_per_type(specs)
    keys = list(runners.keys())
    weights = [max(0.0, float(runners[k]["weight"])) for k in keys]
    total = sum(weights) or 1.0
    probs = [w/total for w in weights]

    def chooser(rng: random.Random) -> str:
        r = rng.random()
        cum = 0.0
        for k, p in zip(keys, probs):
            cum += p
            if r <= cum:
                return k
        return keys[-1]

    return runners, chooser

# ---- Loaders: per-type (deterministic) vs mixture (weighted) ----
def load_runners(specs):
    if MODE == "per_type":
        return load_runners_per_type(specs)
    elif MODE == "weighted":
        return load_runners_weighted(specs)
    else:
        raise ValueError(f"Unknown mode: {MODE}")


def infer_hit(trial: dict) -> bool:
    for k in ("is_success", "is_correct", "grade_correct", "is_identity", "hit", "success"):
        if k in trial:
            return bool(trial[k])
    return False


# ---- Main entrypoint ----
def main():
    # Load Brysbaert norms -> used for concreteness sampling
    load_brysbaert_norms(BRYS_PATH)

    # Always write inside the package's responses/ folder
    base_responses = os.path.join(os.path.dirname(__file__), "responses")
    out_dir = make_run_dir(base=base_responses)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Run dir: {out_dir}")

    combined_path = os.path.join(out_dir, "trials.ndjson")  # all types combined
    latest_path   = os.path.join(out_dir, "latest.json")    # snapshot of last trial

    started_at = datetime.datetime.utcnow().isoformat() + "Z"

    runners = load_runners(ANALOGY_SPECS)   # weights ignored in per-type mode
    keys = list(runners.keys())             # deterministic order = insertion order of ANALOGY_SPECS

    # ------ CONFIG: N trials per type ------
    total_planned = N_TRIALS_PER_TYPE * len(keys)

    # Per-type dirs, paths, counters
    type_dirs   = {k: os.path.join(out_dir, k) for k in keys}
    trial_paths = {k: os.path.join(type_dirs[k], "trials.ndjson") for k in keys}
    for k, d in type_dirs.items():
        os.makedirs(d, exist_ok=True)
        if not os.path.exists(trial_paths[k]):
            open(trial_paths[k], "a", encoding="utf-8").close()
    counts = {k: {"n": 0, "hits": 0} for k in keys}

    summary = None

    # --------- Run N trials per type (deterministic, batched) ---------
    trial_idx = 0
    for kind in keys:  # identity -> cyclic -> pair -> ...
        for j in range(N_TRIALS_PER_TYPE):
            trial_idx += 1
            print(f"\n===== Trial {trial_idx}/{total_planned} [{kind}] ({j+1}/{N_TRIALS_PER_TYPE}) =====")

            # trial = runners[kind]["run"](MODEL, verbose=True)
            trial = runners[kind]["run"](MODEL, verbose=True, **runners[kind].get("kwargs", {}))

            # Write immediately
            append_jsonl(trial_paths[kind], trial)                              # per-type stream
            append_jsonl(combined_path, trial)                                  # run-level stream
            write_summary(os.path.join(type_dirs[kind], "latest.json"), trial)  # per-type snapshot
            write_summary(latest_path, trial)                                   # run-level snapshot

            # Update counters
            counts[kind]["n"] += 1
            counts[kind]["hits"] += int(infer_hit(trial))

            # Per-type summary (kept current)
            type_summary = {
                "model": MODEL,
                "n": counts[kind]["n"],
                "hits": counts[kind]["hits"],
                "success_rate": (counts[kind]["hits"] / counts[kind]["n"]) if counts[kind]["n"] else 0.0,
            }
            write_summary(os.path.join(type_dirs[kind], "summary.json"), type_summary)

            # Run summary (kept current)
            overall_n = sum(v["n"] for v in counts.values())
            overall_h = sum(v["hits"] for v in counts.values())
            summary = {
                "model": MODEL,
                "n_trials_so_far": overall_n,
                "overall_success_rate": (overall_h / overall_n) if overall_n else 0.0,
                "by_type": {
                    t: {
                        "n": counts[t]["n"],
                        "hits": counts[t]["hits"],
                        "success_rate": (counts[t]["hits"] / counts[t]["n"]) if counts[t]["n"] else 0.0
                    } for t in counts
                }
            }
            write_summary(os.path.join(out_dir, "summary.json"), summary)

    # ---- End-of-run record ----
    ended_at = datetime.datetime.utcnow().isoformat() + "Z"
    run_record = {
        "run_id": os.path.basename(out_dir),
        "run_dir": out_dir,
        "model": MODEL,
        "started_at": started_at,
        "ended_at": ended_at,
        "n_trials": summary["n_trials_so_far"] if summary else 0,
        "overall_success_rate": summary["overall_success_rate"] if summary else 0.0,
        "by_type": summary["by_type"] if summary else {},
    }

    runs_index = os.path.join(base_responses, "runs.ndjson")
    append_jsonl(runs_index, run_record)
    write_summary(os.path.join(base_responses, "latest_run.json"), run_record)

    print("\n=== Run complete ===")
    if summary:
        print(json.dumps(summary, indent=2))
    print(f"Saved under {out_dir}")



# python -m analogies.run_analogies
if __name__ == "__main__":
    main()
