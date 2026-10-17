"""python -m analysis.cost [runs_root]   (default: runs/)"""
import glob
import json
import os
import sys

import pandas as pd

from core import paths
from core.load import load_manifests
from analysis.common import save_table

root = sys.argv[1] if len(sys.argv) > 1 else paths.RUNS_DIR
frames = []
for exp in sorted(os.listdir(root)):
    if exp.startswith("_") or not os.path.isdir(os.path.join(root, exp)):
        continue
    m = load_manifests(exp, root)
    frames.append(m[m.status == "ok"].assign(experiment=exp))
d = pd.concat(frames)
c = (d.groupby(["experiment", "task", "agent", "tier"])
       .agg(runs=("runtime_s", "count"), wall_median_s=("runtime_s", "median"),
            cpu_median_s=("cpu_s", "median"), cpu_mean_s=("cpu_s", "mean"))
       .round(2).reset_index())
save_table(c, "compute_cost_by_task", "Per-run wall-clock and CPU time by task; see hardware notes in the text", 4)
mp = glob.glob(os.path.join(root, "*", "*", "manifest.json"))
if mp:
    m = json.load(open(mp[0]))
    print("\nhardware:", {k: m.get(k) for k in ("cpu_model", "cpu_count", "workers", "platform")})
