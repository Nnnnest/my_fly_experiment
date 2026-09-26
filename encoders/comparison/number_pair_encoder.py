"""Encodes a pair of numbers (a, b) as a single ALPN drive vector for the
number-comparison experiment: two disjoint groups of live ALPN indices
(hops=1, same live_alpn_indices.npy used by the bandit/grid-world
experiments), each driven to an intensity proportional to its number's
magnitude -- group A's cells all fire at (a/n_max)*mag, group B's at
(b/n_max)*mag. Uniform per-group driving, same convention as
board_odor_encoder.py's cell groups for tic-tac-toe, but continuous
intensity instead of a fixed +/-1 mark.

Deliberately mode='mb', not 'full': this reuses the SAME validated
step()/train(reward/punish) mechanism that worked for the bandit
(Experiment 1), not the x_add_output/x_teach_output regression path that
failed across all three tiers of the counting experiment. The hypothesis
being tested here is narrower and cleaner: does the mushroom body's
native KC->MBON associative learning handle a graded TWO-INPUT
comparison, the same way it handled a graded SINGLE-input value (bandit),
without going anywhere near the full-brain vision pathway.
"""
import numpy as np


def load_live_alpn(path):
    return np.load(path) if isinstance(path, str) else np.asarray(path)


class NumberPairEncoder:
    def __init__(self, live_indices, n_alpn=685, n_max=9, mag=5.0, seed=0):
        live = load_live_alpn(live_indices)
        rng = np.random.default_rng(seed)
        shuffled = live.copy()
        rng.shuffle(shuffled)
        half = len(shuffled) // 2
        self.group_a = shuffled[:half]
        self.group_b = shuffled[half:]
        self.n_alpn = n_alpn
        self.n_max = n_max
        self.mag = mag

    def encode(self, a, b):
        vec = np.zeros(self.n_alpn, dtype=np.float32)
        vec[self.group_a] = (a / self.n_max) * self.mag
        vec[self.group_b] = (b / self.n_max) * self.mag
        return vec


def sample_pair(rng, n_min=1, n_max=9):
    """Distinct a, b -- no ties, same convention as the bandit's distinct arms."""
    a = int(rng.integers(n_min, n_max + 1))
    b = int(rng.integers(n_min, n_max + 1))
    while b == a:
        b = int(rng.integers(n_min, n_max + 1))
    return a, b
