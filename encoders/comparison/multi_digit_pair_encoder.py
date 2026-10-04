"""General N-digit encoding for the number-comparison experiment --
generalizes two_digit_pair_encoder.py's tens/units split to any number of
digit positions. Each digit POSITION of each number gets its own separate
ALPN group (units, tens, hundreds, ... all disjoint), each driven 0-9 on
the same scale regardless of position. Like the two-digit version, this
means raw total current never gives away the answer (471 and 714 drive
the same total current, same three digits just placed differently) --
solving it requires learning that each position's group matters ~10x more
than the position below it, for every position, not just one.

CAPACITY CONSTRAINT, worth watching: live ALPN positions (hops=1) total
only ~302. Each number needs n_digits separate groups, so both numbers
together need 2*n_digits disjoint groups -- group size shrinks as
n_digits grows (e.g. 4 groups of ~75 cells at n_digits=2, 10 groups of
~30 cells at n_digits=5). If accuracy degrades as n_digits grows, part of
that may be a genuine task-difficulty effect and part may be running out
of per-group encoding capacity -- not fully separable without a deeper
intervention (e.g. hops=2's larger live set), so report group_size
alongside any result instead of attributing degradation to one cause.

Same .encode(a, b) interface as NumberPairEncoder / TwoDigitPairEncoder,
so ConnectomeCompareAgent and the classical baselines work unchanged.
"""
import numpy as np


class MultiDigitPairEncoder:
    def __init__(self, live_indices, n_digits=2, n_alpn=685, digit_max=9, mag=5.0, seed=0):
        if n_digits < 1:
            raise ValueError("n_digits must be >= 1")
        live = np.load(live_indices) if isinstance(live_indices, str) else np.asarray(live_indices)
        rng = np.random.default_rng(seed)
        shuffled = live.copy()
        rng.shuffle(shuffled)

        n_groups = 2 * n_digits
        group_size = len(shuffled) // n_groups
        if group_size < 1:
            raise ValueError(f"n_digits={n_digits} needs {n_groups} groups but only "
                             f"{len(shuffled)} live ALPN positions are available "
                             f"({group_size} cells/group) -- reduce n_digits")
        groups = [shuffled[i * group_size:(i + 1) * group_size] for i in range(n_groups)]
        # groups[0:n_digits] = A's digit positions (0=units,1=tens,...)
        # groups[n_digits:2*n_digits] = B's digit positions, same order
        self.groups_a = groups[:n_digits]
        self.groups_b = groups[n_digits:]
        self.n_digits = n_digits
        self.n_alpn = n_alpn
        self.digit_max = digit_max
        self.mag = mag
        self.group_size = group_size

    def _digits_of(self, n):
        out = []
        for _ in range(self.n_digits):
            out.append(n % 10)
            n //= 10
        return out  # [units, tens, hundreds, ...]

    def encode(self, a, b):
        vec = np.zeros(self.n_alpn, dtype=np.float32)
        for i, d in enumerate(self._digits_of(a)):
            vec[self.groups_a[i]] = (d / self.digit_max) * self.mag
        for i, d in enumerate(self._digits_of(b)):
            vec[self.groups_b[i]] = (d / self.digit_max) * self.mag
        return vec


def sample_n_digit_pair(rng, n_digits):
    """Distinct a, b, both proper n_digits-digit numbers (no leading
    zero -- range [10^(n_digits-1), 10^n_digits - 1], except n_digits=1
    which is [1,9])."""
    lo = 10 ** (n_digits - 1) if n_digits > 1 else 1
    hi = 10 ** n_digits - 1
    a = int(rng.integers(lo, hi + 1))
    b = int(rng.integers(lo, hi + 1))
    while b == a:
        b = int(rng.integers(lo, hi + 1))
    return a, b
