"""Connectome binary-judgment agent for addition: 'is C == A + B?'.

mode='mb', plain step()/train(reward/punish) -- NOT x_teach_output, per
summary_addendum2.txt section 15 (reusing the mechanism validated by the
number-comparison Method A/B experiments, not the x_zone regression
machinery that failed for shape-identity/counting).

calibrate_baseline() must be called once, on a sample of (a,b,c) triples,
BEFORE any train() call -- same discipline as every other agent here.
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_THIS_DIR, os.path.dirname(_THIS_DIR), os.path.dirname(os.path.dirname(_THIS_DIR))):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import numpy as np
from fly_api import FlyBrainAPI
from agents.shared.paths import FLY_ROOT
from encoders.addition.addition_encoder import AdditionEncoder


class ConnectomeAdditionAgent:
    def __init__(self, n_max=9, hops=1, live_indices_path=None, seed=0):
        self.fly = FlyBrainAPI(mode="mb", path=FLY_ROOT)
        live_path = live_indices_path or os.path.join(FLY_ROOT, "my_experiments", "results", "gridworld", "live_alpn_indices.npy")
        self.encoder = AdditionEncoder(live_path, n_max=n_max, seed=seed)
        self.hops = hops
        self._baseline_coef = None

    def calibrate_baseline(self, sample_triples):
        """sample_triples: list of (a, b, c). Call BEFORE any train()."""
        X, y = [], []
        for a, b, c in sample_triples:
            X.append(self.encoder.features(a, b, c))
            raw = self.fly.step(odor=self.encoder.encode(a, b, c), hops=self.hops)["MB_pref"]
            y.append(raw)
        X = np.array(X)
        X_aug = np.hstack([X, np.ones((len(X), 1))])
        coef, *_ = np.linalg.lstsq(X_aug, np.array(y), rcond=None)
        self._baseline_coef = coef

    def _baseline(self, a, b, c):
        if self._baseline_coef is None:
            raise RuntimeError("call calibrate_baseline() before using this agent")
        feats = np.append(self.encoder.features(a, b, c), 1.0)
        return float(feats @ self._baseline_coef)

    def value(self, a, b, c):
        raw = self.fly.step(odor=self.encoder.encode(a, b, c), hops=self.hops)["MB_pref"]
        return raw - self._baseline(a, b, c)

    def predict(self, a, b, c):
        """True = agent believes c == a + b."""
        return self.value(a, b, c) > 0.0

    def update(self, a, b, c, label):
        """label: True if c == a + b, False otherwise."""
        vec = self.encoder.encode(a, b, c)
        if label:
            self.fly.train(odor=vec, reward=1.0, hops=self.hops)
        else:
            self.fly.train(odor=vec, punish=1.0, hops=self.hops)

    def drift_pct(self):
        num = float(np.abs(self.fly.wM - self.fly.wM0).mean())
        den = float(np.abs(self.fly.wM0).mean()) + 1e-9
        return 100.0 * num / den

    def sleep(self, **kwargs):
        return self.fly.sleep(**kwargs)

    def save(self, path_prefix):
        self.fly.save_weights(path_prefix + "_weights.npz")
        if self._baseline_coef is not None:
            np.save(path_prefix + "_baseline.npy", self._baseline_coef)

    def load(self, path_prefix):
        self.fly.load_weights(path_prefix + "_weights.npz")
        self._baseline_coef = np.load(path_prefix + "_baseline.npy")
