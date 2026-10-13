"""python -m analysis.validate   (exit code 1 if anything is missing/failed/inconsistent)"""
import json
import os
import sys

import pandas as pd

from core import paths
from core.config import FINAL_SEEDS
from analysis.common import save_table

EXPERIMENTS = ["bandit", "comparison", "gridworld", "tictactoe", "addition", "addition_diag"]


def read_manifests(exp):
    base = os.path.join(paths.RUNS_DIR, exp)
    rows = []
    for rid in sorted(os.listdir(base)):
        mp = os.path.join(base, rid, "manifest.json")
        if not os.path.exists(mp):
            continue
        with open(mp) as f:
            m = json.load(f)
        rows.append({"experiment": exp, "run_id": rid, "task": m["task"], "agent": m["agent"],
                     "tier": m["tier"], "seed": int(m["seed"]), "status": m["status"],
                     "runtime_s": m.get("runtime_s"), "config_hash": m["config_hash"],
                     "repo_commit": (m.get("repo") or {}).get("commit"),
                     "repo_dirty": (m.get("repo") or {}).get("dirty"),
                     "sim_commit": (m.get("simulator") or {}).get("commit")})
    return pd.DataFrame(rows)


def validate():
    ok, frames = True, []
    for exp in EXPERIMENTS:
        if not os.path.isdir(os.path.join(paths.RUNS_DIR, exp)):
            print(f"[MISSING] runs/{exp}")
            ok = False
            continue
        m = read_manifests(exp)
        frames.append(m)
        print(f"\n== {exp}: {len(m)} runs, status {m.status.value_counts().to_dict()}")
        for _, r in m[m.status != "ok"].iterrows():
            print(f"  [NOT OK] {r.run_id}: {r.status}")
            ok = False
        for key, d in m.groupby(["task", "agent", "tier"]):
            if sorted(d.seed.unique()) != FINAL_SEEDS:
                print(f"  [SEEDS] {key}: {sorted(d.seed.unique())}")
                ok = False
            if d.config_hash.nunique() != 1:
                print(f"  [CONFIG] {key}: {d.config_hash.nunique()} different configs")
                ok = False
        print(f"  repo commits: {m.repo_commit.value_counts().to_dict()}  dirty runs: {int(m.repo_dirty.fillna(False).sum())}")
        print(f"  simulator commits: {m.sim_commit.value_counts().to_dict()}")
    inv = pd.concat(frames, ignore_index=True)
    cost = inv.groupby(["experiment", "agent", "tier"]).agg(
        runs=("runtime_s", "count"), total_h=("runtime_s", lambda s: s.sum() / 3600),
        mean_s=("runtime_s", "mean"), median_s=("runtime_s", "median")).reset_index().round(2)
    save_table(cost, "compute_cost", "Wall-clock cost per run (several workers shared the CPU; compare within an experiment only)", 3)
    print("\nVALIDATION", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    sys.exit(0 if validate() else 1)
