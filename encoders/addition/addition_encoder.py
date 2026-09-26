"""ALPN encoder for the addition experiment (spec.txt rung (b) /
summary_addendum2.txt section 15).

Three disjoint ALPN groups from the live hops=1 set (live_alpn_indices.npy):
operand A, operand B, candidate answer C. Each group is driven to an
intensity proportional to its own value on its own scale (A, B: 0..n_max;
C: 0..2*n_max, since a valid sum can be up to n_max+n_max). Same
intensity-coding trick as the comparison experiments' number_pair_encoder /
two_digit_pair_encoder -- three groups instead of two.
"""
import numpy as np


class AdditionEncoder:
    def __init__(self, live_indices, n_max=9, mag=5.0, n_target=685, seed=0):
        live = np.load(live_indices) if isinstance(live_indices, str) else np.asarray(live_indices)
        live = live.astype(np.int64)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(live))
        third = len(live) // 3
        self.group_a = live[perm[:third]]
        self.group_b = live[perm[third:2 * third]]
        self.group_c = live[perm[2 * third:3 * third]]
        self.n_max = n_max
        self.c_max = 2 * n_max
        self.mag = mag
        self.n_target = n_target

    def encode(self, a, b, c):
        vec = np.zeros(self.n_target, dtype=np.float32)
        vec[self.group_a] = (a / self.n_max) * self.mag
        vec[self.group_b] = (b / self.n_max) * self.mag
        vec[self.group_c] = (c / self.c_max) * self.mag
        return vec

    @staticmethod
    def features(a, b, c):
        """For calibrate_baseline()'s linear regression -- same technique as
        connectome_tictactoe_agent.py, extended with the sum-residual the
        task actually turns on."""
        diff = a + b - c
        return np.array([a, b, c, a * b, a * c, b * c, diff, diff ** 2, a + b],
                        dtype=np.float64)
