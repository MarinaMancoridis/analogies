import os, json, random, importlib, datetime
from analogies.common import make_run_dir, append_jsonl, write_summary

MODEL = "gpt-5"
N_TRIALS = 10
RNG = random.Random()
N_TRIALS_PER_TYPE = 10

# ---- ANALOGY TYPES ----
ANALOGY_SPECS = [
    # {"key": "identity", "module": "analogies.analogy_types.identity", "weight": 1},
    # {"key": "cyclic",   "module": "analogies.analogy_types.cyclic",   "weight": 1},
    {"key": "pair",     "module": "analogies.analogy_types.pair_analogy", "weight": 1},
]
# def load_runners(specs):
#     runners = {}
#     for s in specs:
#         mod = importlib.import_module(s["module"])
#         run = getattr(mod, "run_trial")
#         runners[s["key"]] = {"run": run, "weight": s.get("weight", 1)}
#     return runners

# ---- popular/rare variants of identity ----
ANALOGY_SPECS = [
    {"key": "identity_pop_pop",   "module": "analogies.analogy_types.identity",
     "kwargs": {"A_mode": "popular", "C_mode": "popular"}},
    {"key": "identity_pop_rare",  "module": "analogies.analogy_types.identity",
     "kwargs": {"A_mode": "popular", "C_mode": "rare"}},
    {"key": "identity_rare_pop",  "module": "analogies.analogy_types.identity",
     "kwargs": {"A_mode": "rare",    "C_mode": "popular"}},
    {"key": "identity_rare_rare", "module": "analogies.analogy_types.identity",
     "kwargs": {"A_mode": "rare",    "C_mode": "rare"}},
]


def load_runners(specs):
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

# ---- identity by POS ----
# 10 trials per POS x 5 POS = 50 total runs
POS_LIST = ["noun", "verb", "adjective", "adverb"]
ANALOGY_SPECS = [
    {
        "key": f"identity_pos_{pos}",
        "module": "analogies.analogy_types.identity",
        "kwargs": {"A_mode": f"pos:{pos}", "C_mode": f"pos:{pos}"},
        "weight": 1,
    }
    for pos in POS_LIST
]

def infer_hit(trial: dict) -> bool:
    for k in ("is_success", "is_correct", "grade_correct", "is_identity", "hit", "success"):
        if k in trial:
            return bool(trial[k])
    return False


def main():
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
