"""True two-digit encoding for the number-comparison experiment: each
number is split into its tens digit and units digit, each driving its
OWN separate ALPN group, both scaled 0-9 identically. This is
deliberately NOT just NumberPairEncoder with a wider range -- in that
one-knob-per-number encoding, more total current always means a bigger
number (see number_pair_encoder.py), so a "which side has more current"
heuristic can do most of the work. Here, digits are encoded at equal
per-digit scale, so 47 and 74 drive the SAME total current (4+7 either
way) -- raw intensity no longer gives away the answer. To solve this, the
network has to learn that the tens-digit group matters ~10x more than the
units-digit group: a genuine weighted-combination problem, structurally
closer to the conjunctive patterns that beat the connectome in tic-tac-toe
and digit-ID than to the single-knob comparison that worked well.

4 disjoint ALPN groups total: A's tens, A's units, B's tens, B's units.
Same .encode(a, b) interface as NumberPairEncoder, so
ConnectomeCompareAgent / LinearCompareAgent / MLPCompareAgent all work
unchanged -- only the encoder passed in differs.
"""
import numpy as np


class TwoDigitPairEncoder:
    def __init__(self, live_indices, n_alpn=685, digit_max=9, mag=5.0, seed=0):
        live = np.load(live_indices) if isinstance(live_indices, str) else np.asarray(live_indices)
        rng = np.random.default_rng(seed)
        shuffled = live.copy()
        rng.shuffle(shuffled)
        quarter = len(shuffled) // 4
        self.group_a_tens = shuffled[0 * quarter:1 * quarter]
        self.group_a_units = shuffled[1 * quarter:2 * quarter]
        self.group_b_tens = shuffled[2 * quarter:3 * quarter]
        self.group_b_units = shuffled[3 * quarter:4 * quarter]
        self.n_alpn = n_alpn
        self.digit_max = digit_max
        self.mag = mag

    def encode(self, a, b):
        ta, ua = divmod(a, 10)
        tb, ub = divmod(b, 10)
        vec = np.zeros(self.n_alpn, dtype=np.float32)
        vec[self.group_a_tens] = (ta / self.digit_max) * self.mag
        vec[self.group_a_units] = (ua / self.digit_max) * self.mag
        vec[self.group_b_tens] = (tb / self.digit_max) * self.mag
        vec[self.group_b_units] = (ub / self.digit_max) * self.mag
        return vec


def sample_two_digit_pair(rng, n_min=10, n_max=99):
    """Distinct two-digit a, b -- same no-ties convention as sample_pair."""
    a = int(rng.integers(n_min, n_max + 1))
    b = int(rng.integers(n_min, n_max + 1))
    while b == a:
        b = int(rng.integers(n_min, n_max + 1))
    return a, b
