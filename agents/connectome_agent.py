import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from fly_api import FlyBrainAPI
import numpy as np

class ConnectomeBanditAgent:
    def __init__(self, k, odor_labels, epsilon=0.1, min_pulls=15, seed=0):
        assert k == len(odor_labels)
        self.brain = FlyBrainAPI(mode="mb")
        self.odor_labels = odor_labels
        self.epsilon = epsilon
        self.min_pulls = min_pulls
        self.counts = np.zeros(k)
        self.rng = np.random.default_rng(seed)

        # Fix 1: capture untrained baseline before any train() call
        self.baseline = np.array([
            self.brain.step(odor=o)["MB_pref"] for o in self.odor_labels
        ])

    def _corrected_prefs(self):
        current = np.array([
            self.brain.step(odor=o)["MB_pref"] for o in self.odor_labels
        ])
        return current - self.baseline

    def select_action(self):
        # Fix 2: force every arm to be tried min_pulls times before trusting the network
        under_explored = np.where(self.counts < self.min_pulls)[0]
        if len(under_explored) > 0:
            return int(under_explored[0])
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(self.odor_labels)))
        return int(np.argmax(self._corrected_prefs()))

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.brain.train(odor=self.odor_labels[arm], reward=reward)
