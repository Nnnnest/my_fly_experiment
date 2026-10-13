from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn

from agents.shared.addition_common import sample_batch


def feats(width, kind, a, b, c):
    if width == "single":
        if kind == "raw":
            return [a, b, c]
        d = a + b - c
        return [a, b, c, a * b, a * c, b * c, d, d * d, a + b]
    at, au = divmod(a, 10)
    bt, bu = divmod(b, 10)
    ct, cu = divmod(c, 10)
    if kind == "raw":
        return [at, au, bt, bu, ct, cu]
    us, d = au + bu, a + b - c
    return [at, au, bt, bu, ct, cu, at * bt, au * bu, us, float(us >= 10), d, d * d, a + b]


@lru_cache(maxsize=None)
def _scale(width, kind):
    """Fixed per-feature scale from a deterministic sample (identical for every seed/agent)."""
    two = width == "two"
    rng = np.random.default_rng(12345)
    X = np.array([feats(width, kind, *t[:3]) for t in sample_batch(rng, 49 if two else 9, 4000, two)], float)
    s = np.abs(X).max(0)
    s[s == 0] = 1.0
    return s


def _x(width, kind, a, b, c, scaled=True):
    x = np.array(feats(width, kind, a, b, c), float)
    return x / _scale(width, kind) if scaled else x


class LinearAdditionAgent:
    """Online normalized-LMS toward +1/-1 targets. kind='raw' (inputs only) or 'feat' (engineered
    features incl. sum residual and, for two digits, the carry flag = oracle-feature upper bound).
    scaled: divide features by fixed per-feature maxima (tuned choice)."""

    def __init__(self, width, kind, lr=0.2, seed=0, scaled=True):
        self.width, self.kind, self.lr, self.scaled = width, kind, lr, scaled
        self.w = np.zeros(len(feats(width, kind, 1, 1, 1)) + 1)

    def _v(self, a, b, c):
        return np.append(_x(self.width, self.kind, a, b, c, self.scaled), 1.0)

    def predict(self, a, b, c):
        return float(self._v(a, b, c) @ self.w) > 0.0

    def update(self, a, b, c, label):
        x = self._v(a, b, c)
        err = (1.0 if label else -1.0) - x @ self.w
        self.w += self.lr * err * x / (x @ x + 1e-6)

class MLPAdditionAgent:
    """Dense(64)-Dense(64)-1, BCE, one example per update, same inputs as the linear agent of its kind."""

    def __init__(self, width, kind, lr=1e-3, seed=0):
        torch.manual_seed(seed)
        self.width, self.kind = width, kind
        n = len(feats(width, kind, 1, 1, 1))
        self.net = nn.Sequential(nn.Linear(n, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def _t(self, a, b, c):
        return torch.tensor(_x(self.width, self.kind, a, b, c), dtype=torch.float32).unsqueeze(0)

    def predict(self, a, b, c):
        with torch.no_grad():
            return self.net(self._t(a, b, c)).item() > 0.0

    def update(self, a, b, c, label):
        loss = nn.functional.binary_cross_entropy_with_logits(
            self.net(self._t(a, b, c)), torch.tensor([[1.0 if label else 0.0]]))
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
