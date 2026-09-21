import numpy as np

class MLPBanditAgent:
    def __init__(self, k, lr=0.3, epsilon=0.1, min_pulls=15, seed=0):
        self.rng = np.random.default_rng(seed)
        self.w = self.rng.normal(0, 0.01, size=k)
        self.lr = lr
        self.epsilon = epsilon
        self.min_pulls = min_pulls
        self.counts = np.zeros(k)

    def select_action(self):
        under_explored = np.where(self.counts < self.min_pulls)[0]
        if len(under_explored) > 0:
            return int(under_explored[0])
        if self.rng.random() < self.epsilon:
            return self.rng.integers(len(self.w))
        return int(np.argmax(self.w))

    def update(self, arm, reward):
        self.counts[arm] += 1
        pred = self.w[arm]
        self.w[arm] += self.lr * (reward - pred)
