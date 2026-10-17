"""python -m analysis.cpu_inflation /abs/path/to/sequential/runs
Compares the main (parallel) runs with a sequential rerun of the same run ids."""
import os
import sys

import pandas as pd

from core import paths
from core.load import load_manifests

seq_root = sys.argv[1]
rows = []
for exp in sorted(os.listdir(seq_root)):
    if exp.startswith("_") or not os.path.isdir(os.path.join(seq_root, exp)):
        continue
    a = load_manifests(exp, paths.RUNS_DIR)
    b = load_manifests(exp, seq_root)
    m = a.merge(b, on="run_id", suffixes=("_par", "_seq"))
    m = m[(m.status_par == "ok") & (m.status_seq == "ok")].copy()
    m["experiment"] = exp
    rows.append(m)
d = pd.concat(rows)
d["cpu_ratio"] = d.cpu_s_par / d.cpu_s_seq
d["wall_ratio"] = d.runtime_s_par / d.runtime_s_seq
print(d.groupby(["experiment", "agent_par", "tier_par"])[["cpu_ratio", "wall_ratio"]]
       .median().round(2).to_string())
