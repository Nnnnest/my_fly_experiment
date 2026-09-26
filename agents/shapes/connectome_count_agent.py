"""Connectome circle-counting agent, mode='full'.

Bound to the VIS pool, not KC -- counting_magnitude_diagnostic.py showed
VIS_mean correlates with true count at 0.977 (clean, monotonic), while
ALPN_mean and raw KC activation were both near-zero (expected: ALPN is
the olfactory pool, and per README.md vision only weakly reaches KC even
at hops=3-4). x_add_output's src_pool can be any pool, not just KC (see
its own docstring: "Pass any pool for other circumstances ... ALPN for
innate-like"), so this binds the new "count" output directly to fly.VIS
where the actual graded magnitude signal lives, rather than to KC where
it barely arrives.

Single regression output, trained via x_teach_output's target=float mode
(the library's "value" teaching mode, built for graded magnitude readouts
rather than discrete GO/NOGO classes).
"""
import sys
import os

# Robust path bootstrap: fly_api.py lives at the fly-brain/ repo root, one
# level ABOVE my_experiments/ -- add several ancestor levels rather than
# hard-coding an exact depth, so this survives future directory moves.
_HERE = os.path.dirname(os.path.abspath(__file__))
_d = _HERE
for _ in range(6):
    if _d not in sys.path:
        sys.path.insert(0, _d)
    _d = os.path.dirname(_d)

import numpy as np

from fly_api import FlyBrainAPI
from agents.shared.paths import FLY_ROOT


class ConnectomeCountAgent:
    def __init__(self, n_max=4, n_out=8, hops=3, teach_eta=0.3, seed=0):
        self.fly = FlyBrainAPI(mode="full", path=FLY_ROOT)
        self.fly.enable_scaling()  # build _fan BEFORE x_add_output grows N
        self.n_max = n_max
        self.hops = hops
        self.teach_eta = teach_eta
        self.fly.x_add_output("count", n=n_out, src_pool=self.fly.VIS, seed=seed)
        self.baseline = None

    def calibrate_baseline(self, calib_bitmaps):
        """Untrained X_count reading, averaged over a representative sample
        -- same ordering requirement as the digit agent's calibration:
        call once, before any train_on()."""
        vals = []
        for bitmap in calib_bitmaps:
            out = self.fly.step(image=bitmap, hops=self.hops)
            vals.append(out["X_count"])
        self.baseline = float(np.mean(vals))

    def predict(self, bitmap):
        out = self.fly.step(image=bitmap, hops=self.hops)
        raw = out["X_count"]
        base = self.baseline if self.baseline is not None else 0.0
        est = (raw - base) * self.n_max
        pred = int(round(float(np.clip(est, 0, self.n_max))))
        return pred, raw

    def train_on(self, bitmap, count, trials=3):
        target = count / self.n_max
        self.fly.x_teach_output("count", trials=trials, eta=self.teach_eta,
                                target=target, hops=self.hops, image=bitmap)

    def drift_pct(self):
        num = float(np.abs(self.fly.wM - self.fly.wM0).mean())
        den = float(np.abs(self.fly.wM0).mean()) + 1e-9
        return 100.0 * num / den

    def output_drift_pct(self):
        """Drift restricted to the 'count' output's OWN incoming edges
        (per_in=64 x n_out=8 = 512 edges by default) -- drift_pct() above
        averages over the entire ~15M-edge full-brain graph, so real
        learning at the output can be completely invisible there. This is
        the metric that actually reflects whether x_teach_output is
        moving the weights that matter."""
        ei = self.fly._x_in["count"][0]
        w = self.fly.wM[ei].astype(np.float64)
        w0 = self.fly.wM0[ei].astype(np.float64)
        num = float(np.abs(w - w0).mean())
        den = float(np.abs(w0).mean()) + 1e-9
        return 100.0 * num / den

    def sleep(self, **kwargs):
        return self.fly.sleep(**kwargs)
