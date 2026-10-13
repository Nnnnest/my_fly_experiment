import numpy as np

from core import paths 
from core.brain_pool import fresh_brain, cache_for
from agents.shared.mb_value_cache import train_no_readout


def label_patterns(brain, labels):
    """Vector form of the symbolic odors: 'A' = first half of ALPN, 'B' = second half
    (identical to what fly_api.encode does for odor='A'/'B' in mb mode)."""
    n = len(brain.ALPN)
    half = n // 2
    pats = np.zeros((len(labels), n), dtype=np.float32)
    for i, lab in enumerate(labels):
        if lab == "A":
            pats[i, :half] = 1.0
        elif lab == "B":
            pats[i, half:] = 1.0
        else:
            raise ValueError("only 'A'/'B' labels are supported")
    return pats


class ConnectomeBanditAgent:
    def __init__(self, k, odor_labels, epsilon=0.1, min_pulls=15, seed=0):
        assert k == len(odor_labels)
        self.brain = fresh_brain()
        self.pats = label_patterns(self.brain, odor_labels)
        self.cache = cache_for("bandit:" + ",".join(odor_labels), self.pats)
        self.epsilon, self.min_pulls = epsilon, min_pulls
        self.counts = np.zeros(k)
        self.rng = np.random.default_rng(seed)
        self.baseline = self.cache.values_all().copy()   # untrained MB_pref per arm

    def _corrected_prefs(self):
        return self.cache.values_all() - self.baseline

    def select_action(self):
        under = np.where(self.counts < self.min_pulls)[0]
        if len(under) > 0:
            return int(under[0])
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(self.counts)))
        return int(np.argmax(self._corrected_prefs()))

    def update(self, arm, reward):
        self.counts[arm] += 1
        if reward > 0:   # train() is a no-op for reward == 0 (DAN gate closed)
            train_no_readout(self.brain, self.pats[arm], hops=1, reward=float(reward))
            self.cache.refresh()

    def drift_pct(self):
        num = float(np.abs(self.brain.wM - self.brain.wM0).mean())
        return 100.0 * num / (float(np.abs(self.brain.wM0).mean()) + 1e-9)
