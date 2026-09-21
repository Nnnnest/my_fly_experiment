import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fly_api import FlyBrainAPI
import numpy as np

mb = FlyBrainAPI(mode="mb")
n_target = len(mb.ORN) if len(mb.ORN) else len(mb.ALPN)
print(f"addressable population size: {n_target}")

K = 30  # placeholder state count — adjust once you count traversable cells in your grid
assert K <= n_target, "K exceeds addressable neurons, shrink K or rethink encoding"

def one_hot(i, k=K):
    v = np.zeros(k, dtype=np.float32)
    v[i] = 1.0
    return v

# adjacent indices (states 0 and 1) vs far-apart indices (0 and K-1)
pairs = [(0, 1), (0, K - 1)]
for i, j in pairs:
    vi, vj = one_hot(i), one_hot(j)
    base_i = mb.step(odor=vi)["MB_pref"]
    base_j = mb.step(odor=vj)["MB_pref"]
    mb.train(odor=vi, reward=1.0)
    after_i = mb.step(odor=vi)["MB_pref"]
    after_j = mb.step(odor=vj)["MB_pref"]
    print(f"\nstates {i} vs {j}:")
    print(f"  state {i}: {base_i:.2f} -> {after_i:.2f} (trained, should move)")
    print(f"  state {j}: {base_j:.2f} -> {after_j:.2f} (untrained, ideally stays close to {base_j:.2f})")
    # reset for a clean next pair: fresh instance avoids carrying training across pairs
    mb = FlyBrainAPI(mode="mb")
