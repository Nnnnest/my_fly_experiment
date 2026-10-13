import os
import sys
import numpy as np

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FLY_ROOT = os.path.normpath(os.path.join(REPO_ROOT, ".."))   # fly-brain (fly_api.py, mb_circuit.npz)
ASSETS_DIR = os.path.join(REPO_ROOT, "assets")
RUNS_DIR = os.path.join(REPO_ROOT, "runs")
LIVE_ALPN = os.path.join(ASSETS_DIR, "live_alpn_indices.npy")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if FLY_ROOT not in sys.path:
    sys.path.append(FLY_ROOT)      # append: never shadow local modules


def live_alpn():
    """Live ALPN positions at hops=1 (302). Recomputed analytically if the file is missing.""" 
    if not os.path.exists(LIVE_ALPN):
        os.makedirs(ASSETS_DIR, exist_ok=True)
        from agents.shared.mb_value_cache import ensure_live_hops1
        ensure_live_hops1(LIVE_ALPN)
    return np.load(LIVE_ALPN)
