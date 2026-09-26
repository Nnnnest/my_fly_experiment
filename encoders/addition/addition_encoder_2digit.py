"""Two-digit positional addition encoder (spec.txt rung (b), 'with
carrying' extension). Same trick as two_digit_pair_encoder.py (Method B,
comparison experiment): each number is split into tens/units digits, each
driven on its own ALPN group at the SAME 0-9 scale, so raw total current
no longer gives away the sum -- the agent must weight tens ~10x units and
handle carrying, not just read off total intensity.

Operands a, b are sampled 0..49 so a+b never exceeds 99 (stays two-digit,
no third/hundreds group needed) while still requiring real carrying
whenever units(a)+units(b) >= 10.
"""
import numpy as np


class TwoDigitAdditionEncoder:
    def __init__(self, live_indices, mag=5.0, n_target=685, seed=0):
        live = np.load(live_indices) if isinstance(live_indices, str) else np.asarray(live_indices)
        live = live.astype(np.int64)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(live))
        sixth = len(live) // 6
        names = ["a_t", "a_u", "b_t", "b_u", "c_t", "c_u"]
        self.groups = {}
        for i, name in enumerate(names):
            self.groups[name] = live[perm[i * sixth:(i + 1) * sixth]]
        self.mag = mag
        self.n_target = n_target

    @staticmethod
    def _digits(n):
        return n // 10, n % 10

    def encode(self, a, b, c):
        at, au = self._digits(a)
        bt, bu = self._digits(b)
        ct, cu = self._digits(c)
        vec = np.zeros(self.n_target, dtype=np.float32)
        for name, val in (("a_t", at), ("a_u", au), ("b_t", bt), ("b_u", bu),
                          ("c_t", ct), ("c_u", cu)):
            vec[self.groups[name]] = (val / 9.0) * self.mag
        return vec

    @staticmethod
    def features(a, b, c):
        """Digit-level features for calibrate_baseline() -- includes the
        units-sum (carry trigger: >=10 means a carry is needed) explicitly,
        since that's the one genuinely discrete piece of this task."""
        at, au = TwoDigitAdditionEncoder._digits(a)
        bt, bu = TwoDigitAdditionEncoder._digits(b)
        ct, cu = TwoDigitAdditionEncoder._digits(c)
        units_sum = au + bu
        carry = 1.0 if units_sum >= 10 else 0.0
        diff = a + b - c
        return np.array([at, au, bt, bu, ct, cu, at * bt, au * bu,
                         units_sum, carry, diff, diff ** 2, a + b],
                        dtype=np.float64)
