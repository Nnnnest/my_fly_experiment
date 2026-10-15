"""python -m analysis.compare_runs runs_pre_freeze"""
import json
import os
import sys

from core import paths

old_root = os.path.join(paths.REPO_ROOT, sys.argv[1])
for exp in sorted(os.listdir(paths.RUNS_DIR)):
    if exp.startswith("_") or not os.path.isdir(os.path.join(old_root, exp)):
        continue
    n = same = 0
    worst = []
    for rid in sorted(os.listdir(os.path.join(paths.RUNS_DIR, exp))):
        po = os.path.join(old_root, exp, rid, "summary.json")
        pn = os.path.join(paths.RUNS_DIR, exp, rid, "summary.json")
        if not (os.path.exists(po) and os.path.exists(pn)):
            continue
        a, b = json.load(open(po)), json.load(open(pn))
        num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
        d = max([abs(a[k] - b[k]) for k in a if k in b and num(a[k]) and num(b[k])] or [0.0])
        n += 1
        same += d < 1e-9
        worst.append((d, rid))
    worst.sort(reverse=True)
    print(f"{exp}: compared {n}, identical {same}, max diff {worst[0][0]:.4g}" if worst else f"{exp}: nothing to compare")
    for d, rid in worst[:3]:
        print(f"    {rid}: {d:.4g}")
