"""MultiDigitPairEncoder with a fixed number of ALPNs per digit position (capacity-matched control).

MultiDigitPairEncoder splits the ~302 live ALPNs into 2*n_digits groups, so the group size falls from 75 (2 digits) to
16 (9 digits): digit count and per-position capacity change together. Here every position gets `per_position` ALPNs,
independent of n_digits. Same shuffle, same group layout, same encode(); with per_position equal to the original group
size it is identical to MultiDigitPairEncoder."""
import numpy as np

from encoders.comparison.multi_digit_pair_encoder import MultiDigitPairEncoder


class CapacityDigitPairEncoder(MultiDigitPairEncoder):
    def __init__(self, live_indices, n_digits=2, per_position=16, n_alpn=685, digit_max=9, mag=5.0, seed=0):
        live = np.load(live_indices) if isinstance(live_indices, str) else np.asarray(live_indices)
        rng = np.random.default_rng(seed)
        shuffled = live.copy()
        rng.shuffle(shuffled)
        n_groups = 2 * n_digits
        if n_groups * per_position > len(shuffled):
            raise ValueError(f"{n_groups} groups x {per_position} ALPNs exceed the {len(shuffled)} live positions")
        groups = [shuffled[i * per_position:(i + 1) * per_position] for i in range(n_groups)]
        self.groups_a, self.groups_b = groups[:n_digits], groups[n_digits:]
        self.n_digits, self.n_alpn, self.digit_max, self.mag = n_digits, n_alpn, digit_max, mag
        self.group_size = per_position
