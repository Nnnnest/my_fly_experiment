"""Connectome number-comparison agent, mode='mb' -- standard step()/
train(reward/punish), no x_zone. MB_pref's sign is the decision: >=
baseline => "A is bigger", < baseline => "B is bigger" (baseline-
corrected the same way the bandit corrected for innate odor bias --
group_a/group_b are not guaranteed to read as neutral before any
training, so calibrate_baseline() must run once, before any train_on()
call, same ordering requirement as every other calibrate_baseline() in
this project).

train_on() reinforces the CORRECT reading of the current pair's pattern:
reward=1.0 if a>b (push this pattern's readout toward approach/positive),
punish=1.0 if b>a (push toward avoid/negative) -- exactly the bandit's
sign-gate convention, just applied per-pair instead of per-arm.
"""
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_d = _HERE
for _ in range(4):
    if _d not in sys.path:
        sys.path.insert(0, _d)
    _d = os.path.dirname(_d)

import numpy as np

from fly_api import FlyBrainAPI
from agents.shared.paths import FLY_ROOT

class ConnectomeCompareAgent:
    def __init__(self, encoder, hops=1):
        self.fly = FlyBrainAPI(mode="mb", path=FLY_ROOT)
        self.encoder = encoder
        self.hops = hops
        self.baseline = None  # per-pair baseline model, set by calibrate_baseline()
        self._coef = None

    def calibrate_baseline(self, calib_pairs):
        """Fit a linear baseline MB_pref(a, b) ~ a + b + a*b on untrained
        readouts, same reasoning as connectome_tictactoe_agent.py's
        calibrate_baseline: a single global constant wouldn't capture that
        the innate bias may itself depend on a and b, and whole-graph
        plasticity means a lazy per-pair cache would drift as training on
        OTHER pairs proceeds. Call once, before any train_on()."""
        X, y = [], []
        for a, b in calib_pairs:
            vec = self.encoder.encode(a, b)
            pref = self.fly.step(odor=vec, hops=self.hops)["MB_pref"]
            X.append([a, b, a * b, 1.0])
            y.append(pref)
        X = np.array(X)
        coef, *_ = np.linalg.lstsq(X, np.array(y), rcond=None)
        self._coef = coef

    def _baseline(self, a, b):
        if self._coef is None:
            raise RuntimeError("call calibrate_baseline() before predict()/train_on()")
        return float(np.array([a, b, a * b, 1.0]) @ self._coef)

    def predict(self, a, b):
        vec = self.encoder.encode(a, b)
        pref = self.fly.step(odor=vec, hops=self.hops)["MB_pref"]
        corrected = pref - self._baseline(a, b)
        return ("A" if corrected >= 0 else "B"), corrected

    def train_on(self, a, b):
        vec = self.encoder.encode(a, b)
        if a > b:
            self.fly.train(odor=vec, reward=1.0, hops=self.hops)
        else:
            self.fly.train(odor=vec, punish=1.0, hops=self.hops)

    def drift_pct(self):
        """Standard whole-wM drift is meaningful again here (unlike the
        full-mode counting agent): mb mode's KC->MBON edge count is small
        (~tens of thousands, not 15M), so it isn't diluted the same way."""
        num = float(np.abs(self.fly.wM - self.fly.wM0).mean())
        den = float(np.abs(self.fly.wM0).mean()) + 1e-9
        return 100.0 * num / den

    def sleep(self, **kwargs):
        return self.fly.sleep(**kwargs)
