import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fly_api import FlyBrainAPI
import numpy as np

N_TARGET = 685  # len(mb.ALPN)
K = 30           # placeholder state count, adjust to your actual traversable-cell count

def one_hot(i, k=K, n_target=N_TARGET, mag=1.0):
    v = np.zeros(n_target, dtype=np.float32)
    v[i] = mag
    return v

print("magnitude=1.0 (what we just tested, scaled by encode()'s internal *2.0):")
mb = FlyBrainAPI(mode="mb")
zero_count = 0
for i in range(K):
    pref = mb.step(odor=one_hot(i))["MB_pref"]
    is_zero = abs(pref) < 1e-9
    zero_count += is_zero
    print(f"  state {i:2d}: MB_pref={pref:8.4f}{'  <- exactly zero' if is_zero else ''}")
print(f"{zero_count}/{K} states produced exactly zero\n")

print("magnitude=5.0 (stronger drive, same K states):")
mb2 = FlyBrainAPI(mode="mb")
zero_count2 = 0
for i in range(K):
    pref = mb2.step(odor=one_hot(i, mag=5.0))["MB_pref"]
    is_zero = abs(pref) < 1e-9
    zero_count2 += is_zero
    print(f"  state {i:2d}: MB_pref={pref:8.4f}{'  <- exactly zero' if is_zero else ''}")
print(f"{zero_count2}/{K} states produced exactly zero")
