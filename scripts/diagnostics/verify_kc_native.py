"""apply_rule() must reproduce FlyBrainAPI.train() bit for bit, and the cache-based KC activity must match the simulator's
forward pass (including the DAN drive of the US pathway). Also prints what a zero input does to the plasticity mask.
    python -m scripts.diagnostics.verify_kc_native"""
import numpy as np

from core import paths
from core.brain_pool import fresh_brain, value_fn
from agents.shared.mb_value_cache import train_no_readout
from agents.shared.kc_native import train_inputs, apply_rule, mask_stats, kc_activity

rng = np.random.default_rng(0)
live = paths.live_alpn()
b = fresh_brain()
n = len(b.ALPN)


def rand_vec(zero=False):
    v = np.zeros(n, np.float32)
    if not zero:
        idx = rng.choice(live, size=int(rng.integers(1, 60)), replace=False)
        v[idx] = rng.uniform(0.2, 5.0, len(idx))
    return v


seq = [(rand_vec(zero=i in (0, 7, 31)), bool(rng.random() < 0.5)) for i in range(100)]
b = fresh_brain()
for v, lab in seq:
    train_no_readout(b, v, hops=1, **({"reward": 1.0} if lab else {"punish": 1.0}))
w_native = b.wM.copy()
b = fresh_brain()
cache, _ = value_fn()
err = 0.0
for v, lab in seq:
    kcs, m, s = train_inputs(b, v, lab)
    err = max(err, float(np.abs(kc_activity(cache, v)[0] - kcs).max()))
    apply_rule(b, kcs, m, s, lab)
print("max |w_train - w_apply_rule| =", float(np.abs(w_native - b.wM).max()), "(want 0)")
print("max |KC activity: cache - simulator forward| =", err, "(0 means the US drive does not reach the KCs)")
b = fresh_brain()
kcs, m, s = train_inputs(b, np.zeros(n, np.float32), True)
st = mask_stats(kcs)
print("zero input:", {k: (v if np.isscalar(v) else int(v[0])) for k, v in st.items()})
cnt = np.zeros(len(b.KC))
for v, lab in seq[:20]:
    kcs, m, s = train_inputs(b, v, lab)
    apply_rule(b, kcs, m, s, lab, renorm=False, push_pull=True, decay_n0=10, counts=cnt, mask_active=True)
print("relaxed variants run, weights finite:", bool(np.isfinite(b.wM).all()))
assert np.abs(w_native - fresh_brain().wM).max() > 0
