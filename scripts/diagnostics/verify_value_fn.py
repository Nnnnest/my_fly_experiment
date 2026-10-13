"""value_fn must equal brain.step MB_pref for arbitrary vectors. python -m scripts.diagnostics.verify_value_fn"""
import numpy as np
from core import paths
from core.brain_pool import fresh_brain, value_fn
from encoders.comparison.multi_digit_pair_encoder import MultiDigitPairEncoder

b = fresh_brain()
c, f = value_fn()
enc = MultiDigitPairEncoder(paths.live_alpn(), n_digits=3, seed=0)
rng = np.random.default_rng(0)
errs = []
for _ in range(10):
    a, bb = int(rng.integers(100, 999)), int(rng.integers(100, 999))
    v = enc.encode(a, bb)
    errs.append(abs(float(f(v)[0]) - b.step(odor=v, hops=1)["MB_pref"]))
print("max abs err:", max(errs), "(want < 1e-3)")
