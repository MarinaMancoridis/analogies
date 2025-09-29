import os, json, random, importlib, datetime  # <-- add datetime
from analogies.common import make_run_dir, append_jsonl, write_summary

MODEL = "gpt-5"
N_TRIALS = 10
RNG = random.Random()

ANALOGY_SPECS = [
    {"key": "identity", "module": "analogies.analogy_types.identity", "weight": 1},
    {"key": "cyclic",   "module": "analogies.analogy_types.cyclic",   "weight": 1},
]

def load_runners(specs):
    runners = {}
    for s in specs:
        mod = importlib.import_module(s["module"])
        run = getattr(mod, "run_trial")
        runners[s["key"]] = {"run": run, "weight": s.get("weight", 1)}
    return runners

def infer_hit(trial: dict) -> bool:
    for k in ("is_success", "is_correct", "grade_correct", "is_identity", "hit", "success"):
        if k in trial:
            return bool(trial[k])
    return False

def main():
    out_dir = make_run_dir()
    print(f"Run dir: {out_dir}")

    # ---- start timestamp for this run ----
    started_at = datetime.datetime.utcnow().isoformat() + "Z"  # <-- add

    runners = load_runners(ANALOGY_SPECS)
    keys = list(runners.keys())
    weights = [runners[k]["weight"] for k in keys]

    type_dirs   = {k: os.path.join(out_dir, k) for k in keys}
    trial_paths = {k: os.path.join(type_dirs[k], "trials.ndjson") for k in keys}
    for d in type_dirs.values(): os.makedirs(d, exist_ok=True)
    counts = {k: {"n": 0, "hits": 0} for k in keys}

    for i in range(N_TRIALS):
        kind = random.choices(keys, weights=weights, k=1)[0]
        print(f"\n===== Trial {i+1}/{N_TRIALS} [{kind}] =====")

        trial = runners[kind]["run"](MODEL, verbose=True)
        append_jsonl(trial_paths[kind], trial)

        counts[kind]["n"] += 1
        counts[kind]["hits"] += int(infer_hit(trial))

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

    # ---- end timestamp + append a one-line record for the whole run ----
    ended_at = datetime.datetime.utcnow().isoformat() + "Z"  # <-- add

    # Build the run record using final summary from above loop
    run_record = {
        "run_id": os.path.basename(out_dir),
        "run_dir": out_dir,
        "model": MODEL,
        "started_at": started_at,
        "ended_at": ended_at,
        # snapshot of final summary
        "n_trials": summary["n_trials_so_far"],
        "overall_success_rate": summary["overall_success_rate"],
        "by_type": summary["by_type"],
    }

    responses_root = os.path.dirname(out_dir)                     # e.g., "responses"
    runs_index = os.path.join(responses_root, "runs.ndjson")      # <-- append-only log
    append_jsonl(runs_index, run_record)                          # <-- add

    # optional: a quick pointer to the last run
    write_summary(os.path.join(responses_root, "latest_run.json"), run_record)  # <-- add

    print("\n=== Run complete ===")
    print(json.dumps(summary, indent=2))
    print(f"Saved under {out_dir}")

if __name__ == "__main__":
    main()
