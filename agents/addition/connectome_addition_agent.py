import numpy as np

from core import paths  # noqa: F401
from core.brain_pool import fresh_brain, value_fn
from agents.shared.mb_value_cache import train_no_readout


class ConnectomeAdditionAgent:
    """'is C == A + B?'  value > 0 => yes. Baseline correction uses LABEL-BLIND features only
    (operands, candidate, pairwise products); no sum residual, no carry flag."""

    def __init__(self, encoder, width):
        self.brain = fresh_brain()
        self.cache, self._values = value_fn()
        self.encoder, self.width, self._coef = encoder, width, None

    def _blind(self, a, b, c):
        if self.width == "single":
            return [a, b, c, a * b, a * c, b * c, 1.0]
        at, au = divmod(a, 10)
        bt, bu = divmod(b, 10)
        ct, cu = divmod(c, 10)
        return [at, au, bt, bu, ct, cu, at * bt, au * bu, 1.0]

    def _raw_many(self, triples):
        return self._values(np.stack([self.encoder.encode(a, b, c) for a, b, c in triples]))

    def calibrate_baseline(self, triples):
        """Fit untrained MB_pref ~ blind features. Call once, before any update()."""
        X = np.array([self._blind(*t) for t in triples], float)
        self._coef, *_ = np.linalg.lstsq(X, self._raw_many(triples), rcond=None)

    def value_many(self, triples):
        X = np.array([self._blind(*t) for t in triples], float)
        return self._raw_many(triples) - X @ self._coef

    def value(self, a, b, c):
        return float(self.value_many([(a, b, c)])[0])

    def predict_many(self, triples):
        return self.value_many(triples) > 0.0

    def predict(self, a, b, c):
        return self.value(a, b, c) > 0.0

    def update(self, a, b, c, label):
        kw = {"reward": 1.0} if label else {"punish": 1.0}
        train_no_readout(self.brain, self.encoder.encode(a, b, c), hops=1, **kw)
        self.cache.refresh()

    def drift_pct(self):
        num = float(np.abs(self.brain.wM - self.brain.wM0).mean())
        return 100.0 * num / (float(np.abs(self.brain.wM0).mean()) + 1e-9)

    def sleep(self, **kw):
        r = self.brain.sleep(**kw)
        self.cache.refresh()
        return r
