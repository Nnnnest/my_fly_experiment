import os
import sys
import numpy as np

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FLY_ROOT = os.path.normpath(os.path.join(REPO_ROOT, ".."))   # fly-brain (fly_api.py, mb_circuit.npz)
ASSETS_DIR = os.path.join(REPO_ROOT, "assets")
RUNS_DIR = os.environ.get("FLY_RUNS_DIR") or os.path.join(REPO_ROOT, "runs")
# wiring control: FLY_CIRCUIT_DIR = directory holding a modified mb_circuit.npz (+ mb_groups*.npz)
CIRCUIT_ROOT = os.environ.get("FLY_CIRCUIT_DIR") or FLY_ROOT
_LIVE_NAME = ("live_alpn_indices.npy" if CIRCUIT_ROOT == FLY_ROOT else
              f"live_alpn_indices_{os.path.basename(os.path.normpath(CIRCUIT_ROOT))}.npy")
LIVE_ALPN = os.path.join(ASSETS_DIR, _LIVE_NAME)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if FLY_ROOT not in sys.path:
    sys.path.append(FLY_ROOT)      # append: never shadow local modules


def live_alpn():
    """Live ALPN positions at hops=1 of the ACTIVE circuit (302 for the real one). Computed if the file is missing."""
    if not os.path.exists(LIVE_ALPN):
        os.makedirs(ASSETS_DIR, exist_ok=True)
        from fly_api import FlyBrainAPI
        from agents.shared.mb_value_cache import ensure_live_hops1
        ensure_live_hops1(LIVE_ALPN, FlyBrainAPI(mode="mb", path=CIRCUIT_ROOT))
    return np.load(LIVE_ALPN)
