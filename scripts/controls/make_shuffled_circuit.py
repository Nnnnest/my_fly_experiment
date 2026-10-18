"""python -m scripts.controls.make_shuffled_circuit --seed 0
Degree-preserving shuffle of the ALPN->KC wiring: the postsynaptic endpoints of all ALPN->KC edges are
permuted among those edges. Every ALPN keeps its out-degree and out-weights, every KC keeps its in-degree.
KC->MBON, DAN and all other edges are untouched."""
import argparse
import os
import shutil

import numpy as np

from core import paths

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

src = paths.FLY_ROOT
dst = os.path.join(paths.ASSETS_DIR, "circuits", f"shuffled_s{a.seed}")
os.makedirs(dst, exist_ok=True)
d = dict(np.load(os.path.join(src, "mb_circuit.npz")))
m = np.isin(d["pre"], d["inputs_ALPN"]) & np.isin(d["post"], d["KC"])
post = d["post"].copy()
post[m] = np.random.default_rng(a.seed).permutation(post[m])
d["post"] = post
np.savez(os.path.join(dst, "mb_circuit.npz"), **d)
for f in os.listdir(src):
    if f.startswith("mb_groups") and f.endswith(".npz"):
        shutil.copy(os.path.join(src, f), dst)
print(f"wrote {dst}: shuffled {int(m.sum())} ALPN->KC edges of {len(post)}")
