import numpy as np

from core import paths
from core.brain_pool import fresh_brain, value_fn
from agents.shared.mb_value_cache import train_no_readout


class ConnectomeCompareAgent:
    """MB_pref (baseline-corrected) > 0  =>  'A is bigger'. Values via exact cache, training via train()."""

    def __init__(self, encoder):
        self.brain = fresh_brain()
        self.cache, self._values = value_fn()
        self.encoder = encoder
        self._coef = None

    def _raw(self, a, b):
        return float(self._values(self.encoder.encode(a, b))[0])

    def calibrate_baseline(self, calib_pairs):
        """Fit raw ~ a + b + a*b + 1 on UNTRAINED readouts. Call once, before any train_on()."""
        X = np.array([[a, b, a * b, 1.0] for a, b in calib_pairs])
        y = np.array([self._raw(a, b) for a, b in calib_pairs])
        self._coef, *_ = np.linalg.lstsq(X, y, rcond=None)

    def predict(self, a, b):
        base = float(np.array([a, b, a * b, 1.0]) @ self._coef)
        corrected = self._raw(a, b) - base
        return ("A" if corrected >= 0 else "B"), corrected

    def train_on(self, a, b):
        vec = self.encoder.encode(a, b)
        kw = {"reward": 1.0} if a > b else {"punish": 1.0}
        train_no_readout(self.brain, vec, hops=1, **kw)
        self.cache.refresh()

    def drift_pct(self):
        num = float(np.abs(self.brain.wM - self.brain.wM0).mean())
        return 100.0 * num / (float(np.abs(self.brain.wM0).mean()) + 1e-9)

    def sleep(self, **kw):
        r = self.brain.sleep(**kw)
        self.cache.refresh()
        return r
