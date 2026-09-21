import numpy as np

class QLearningBanditAgent:
    def __init__(self, k, epsilon=0.1, min_pulls=15, seed=0):
        self.q = np.zeros(k)
        self.counts = np.zeros(k)
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.min_pulls = min_pulls

    def select_action(self):
        under_explored = np.where(self.counts < self.min_pulls)[0]
        if len(under_explored) > 0:
            return int(under_explored[0])
        if self.rng.random() < self.epsilon:
            return self.rng.integers(len(self.q))
        return int(np.argmax(self.q))

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.q[arm] += (reward - self.q[arm]) / self.counts[arm]
