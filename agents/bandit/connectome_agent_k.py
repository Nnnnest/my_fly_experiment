import numpy as np

from core import paths  # noqa: F401
from core.brain_pool import fresh_brain, cache_for
from agents.shared.mb_value_cache import train_no_readout


def group_patterns(brain, k):
    """Arm i drives the i-th contiguous 1/k block of the ALPN pool (k=2 reproduces the 'A'/'B' halves)."""
    n = len(brain.ALPN)
    pats = np.zeros((k, n), dtype=np.float32)
    for i in range(k):
        pats[i, i * n // k:(i + 1) * n // k] = 1.0
    return pats


class ConnectomeBanditAgentK:
    def __init__(self, k, epsilon=0.1, min_pulls=15, seed=0):
        self.brain = fresh_brain()
        self.pats = group_patterns(self.brain, k)
        self.cache = cache_for(f"bandit_k{k}", self.pats)
        self.epsilon, self.min_pulls = epsilon, min_pulls
        self.counts = np.zeros(k)
        self.rng = np.random.default_rng(seed)
        self.baseline = self.cache.values_all().copy()

    def select_action(self):
        under = np.where(self.counts < self.min_pulls)[0]
        if len(under) > 0:
            return int(under[0])
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(self.counts)))
        return int(np.argmax(self.cache.values_all() - self.baseline))

    def update(self, arm, reward):
        self.counts[arm] += 1
        if reward > 0:
            train_no_readout(self.brain, self.pats[arm], hops=1, reward=float(reward))
            self.cache.refresh()

    def drift_pct(self):
        num = float(np.abs(self.brain.wM - self.brain.wM0).mean())
        return 100.0 * num / (float(np.abs(self.brain.wM0).mean()) + 1e-9)
