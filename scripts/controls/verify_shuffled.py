"""python -m scripts.controls.verify_shuffled --seed 0"""
import argparse
import os

import numpy as np

from core import paths

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
real = np.load(os.path.join(paths.FLY_ROOT, "mb_circuit.npz"))
shuf = np.load(os.path.join(paths.ASSETS_DIR, "circuits", f"shuffled_s{a.seed}", "mb_circuit.npz"))


def ak(d):
    return np.isin(d["pre"], d["inputs_ALPN"]) & np.isin(d["post"], d["KC"])


ma, mb = ak(real), ak(shuf)
print("edges:", len(real["pre"]), len(shuf["pre"]), "| ALPN->KC edges:", int(ma.sum()), int(mb.sum()))
print("same ALPN->KC edge set (pre unchanged):", bool(np.array_equal(ma, mb)))
print("all other edges, pre and weights identical:",
      bool(np.array_equal(real["pre"], shuf["pre"]) and np.array_equal(real["weight"], shuf["weight"])
           and np.array_equal(real["post"][~ma], shuf["post"][~ma])))
print("ALPN out-degree preserved:", bool(np.array_equal(np.bincount(real["pre"][ma]), np.bincount(shuf["pre"][mb]))))
print("KC in-degree preserved:", bool(np.array_equal(np.bincount(real["post"][ma]), np.bincount(shuf["post"][mb]))))
print("fraction of ALPN->KC edges that changed endpoint:", float((real["post"][ma] != shuf["post"][mb]).mean()))
