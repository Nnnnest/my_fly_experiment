"""Reused+reset brain with cache must reproduce a fresh FlyBrainAPI with plain train().
Run: python -m scripts.diagnostics.verify_brain_reset"""
import numpy as np

from core import paths
from fly_api import FlyBrainAPI
from core.paths import FLY_ROOT
from core.brain_pool import fresh_brain, cache_for
from agents.shared.mb_value_cache import train_no_readout
from agents.bandit.connectome_agent import label_patterns

rng = np.random.default_rng(0)
seq = [(int(rng.integers(2)), float(rng.random() < 0.5)) for _ in range(80)]
labels = ["A", "B"]

ref = FlyBrainAPI(mode="mb", path=FLY_ROOT)
for arm, r in seq:
    ref.train(odor=labels[arm], reward=r)          # original behaviour, incl. reward=0 calls


def run_pooled():
    b = fresh_brain()
    pats = label_patterns(b, labels)
    c = cache_for("verify", pats)
    for arm, r in seq:
        if r > 0:
            train_no_readout(b, pats[arm], hops=1, reward=r)
            c.refresh()
    return b, c


b, c = run_pooled()
print("max |wM_ref - wM_pooled| =", float(np.abs(ref.wM - b.wM).max()), "(want < 1e-4)")
print("cache vs step MB_pref:", [(float(c.values_all()[i]), ref.step(odor=labels[i])["MB_pref"]) for i in range(2)])

w_first = b.wM.copy()
b2 = fresh_brain()
print("after reset, wM == wM0:", bool(np.array_equal(b2.wM, b2.wM0)))
b3, _ = run_pooled()
print("second pass identical to first:", bool(np.array_equal(w_first, b3.wM)))
