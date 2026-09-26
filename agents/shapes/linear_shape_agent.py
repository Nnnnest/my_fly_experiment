"""Lowest-capacity classical baseline for the shape-association experiment.

Grid-world/tic-tac-toe used tabular Q-learning as the "tabular" baseline,
but there's no natural tabular analogue for continuous pixel input, so
this single linear layer + softmax (multinomial logistic regression)
fills that role instead -- flagged explicitly here as a substitution,
same as the Tier-2 gate asymmetry was flagged in summary_addendum.txt.
"""
import numpy as np


class LinearShapeAgent:
    def __init__(self, n_pixels, n_classes=5, lr=0.05, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.01, size=(n_pixels, n_classes))
        self.b = np.zeros(n_classes)
        self.lr = lr

    def _probs(self, x):
        z = x @ self.W + self.b
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def predict(self, bitmap):
        x = bitmap.reshape(-1)
        p = self._probs(x)
        return int(np.argmax(p)), p

    def train_on(self, bitmap, label):
        x = bitmap.reshape(-1)
        p = self._probs(x)
        y = np.zeros_like(p)
        y[label] = 1.0
        err = p - y
        self.W -= self.lr * np.outer(x, err)
        self.b -= self.lr * err
