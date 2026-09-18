import numpy as np

class BernoulliBandit:
    """K-armed bandit, reward is 1 with probability p[arm], else 0."""
    def __init__(self, probs, seed=0):
        self.probs = np.array(probs)
        self.k = len(probs)
        self.rng = np.random.default_rng(seed)

    def reset(self):
        return None  # no state — classic bandit has no observation

    def pull(self, arm):
        reward = float(self.rng.random() < self.probs[arm])
        return reward
