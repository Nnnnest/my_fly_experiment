import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fly_api import FlyBrainAPI
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--hops", type=int, default=1)
parser.add_argument("--mag", type=float, default=5.0)
args = parser.parse_args()

mb = FlyBrainAPI(mode="mb")
n_target = len(mb.ALPN)

def one_hot(i, n_target=n_target, mag=args.mag):
    v = np.zeros(n_target, dtype=np.float32)
    v[i] = mag
    return v

live = []
for i in range(n_target):
    pref = mb.step(odor=one_hot(i), hops=args.hops)["MB_pref"]
    if abs(pref) > 1e-9:
        live.append(i)

print(f"{len(live)}/{n_target} ALPN indices produce nonzero MB_pref at hops={args.hops}")
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"live_alpn_indices_hops{args.hops}.npy")
np.save(out_path, np.array(live))
print("saved to", out_path)
