"""python -m analysis.wiring_tables   (after scripts/run_wiring_control.sh; do NOT set FLY_RUNS_DIR)"""
import collections
import glob
import json
import os

import pandas as pd

from core import paths
from analysis.common import load, cells, wide, paired_table, save_table

CIRCUITS = {"real": paths.RUNS_DIR}
for s in (0, 1, 2):
    CIRCUITS[f"shuf{s}"] = os.path.join(paths.REPO_ROOT, f"runs_shuffled_s{s}")
SPEC = [("bandit", "cum_pseudo_regret", "-"), ("comparison", "eval_acc_final", "fair"),
        ("addition", "eval_acc_final", "fair")]


def status(root, exp):
    c = collections.Counter()
    for p in glob.glob(os.path.join(root, exp, "*", "manifest.json")):
        m = json.load(open(p))
        c[(m["status"], (m["repo"]["commit"] or "")[:8], m["repo"]["dirty"])] += 1
    return dict(c)


def gather(exp, agent, tier):
    frames = []
    for name, root in CIRCUITS.items():
        if not os.path.isdir(os.path.join(root, exp)):
            continue
        print(f"{exp} [{name}] status/commit/dirty:", status(root, exp))
        d = load(exp, root=root)
        d = d[(d.agent == agent) & (d.tier == tier)].copy()
        d["tier"] = name
        frames.append(d)
    return pd.concat(frames)


for exp, met, tier in SPEC:
    d = gather(exp, "connectome", tier)
    c = cells(d, ["task", "tier"], [met], 3)
    save_table(wide(c, "task", met, col="tier"), f"wiring_{exp}",
               f"{exp}: {met}, connectome on the real vs degree-preserving shuffled ALPN->KC wiring (3 shuffles)")
    comps = [(f"real - {n}", ("connectome", "real"), ("connectome", n)) for n in CIRCUITS if n != "real"]
    save_table(paired_table(d, met, comps), f"wiring_{exp}_paired", f"{exp}: paired real - shuffled (matched on task, seed)", 3)

frames = []
for name, root in CIRCUITS.items():
    if os.path.isdir(os.path.join(root, "addition_diag")):
        g = load("addition_diag", root=root)
        g["circuit"] = name
        frames.append(g)
g = pd.concat(frames)
cols = ["jaccard_diff+1", "jaccard_diff-1", "jaccard_random_mean"]
save_table(cells(g, ["task", "circuit"], cols, 3), "wiring_kc_overlap",
           "KC-code Jaccard overlap of the correct addition triple vs c off by +-1, and random baseline", 2)
