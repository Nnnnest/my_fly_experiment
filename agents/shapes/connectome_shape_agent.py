"""Connectome shape-association agent, mode='full' (real photoreceptors --
mb mode's image path is a random projection, not usable for a meaningful
shape-recognition result, per fly_api.py's own comment on IMG_N/_img_proj).

One x_add_output() readout per digit class, bound to the KC pool (the
library's own recommended default src_pool -- "the associative code, so
the output binds stimulus conjunctions"). predict() = argmax over the
X_digit_<n> readouts. train_on() teaches GO on the correct digit's output
and NOGO on one randomly chosen wrong digit's output (mirrors the
positive/one-negative pattern already used in
train_imitation_tictactoe.py, rather than NOGO-ing all wrong classes
every round -- keeps the per-example train() cost from scaling with
n_digits).
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import numpy as np

from fly_api import FlyBrainAPI
from agents.shared.paths import FLY_ROOT

N_DIGITS = 5


class ConnectomeShapeAgent:
    def __init__(self, digits=range(N_DIGITS), n_out=8, teach_eta=0.3, seed=0):
        self.fly = FlyBrainAPI(mode="full", path=FLY_ROOT)
        self.fly.enable_scaling()
        self.digits = list(digits)
        self.teach_eta = teach_eta
        self.rng = np.random.default_rng(seed)
        self.baseline = None
        for d in self.digits:
            self.fly.x_add_output(f"digit_{d}", n=n_out, src_pool=self.fly.KC, seed=seed + d)

    def calibrate_baseline(self, calib_bitmaps):
        """Call once, BEFORE any train_on() call, with a representative sample
        of bitmaps per digit (see run_shapes.py). Measures each output's
        untrained activity level so predict() compares apples to apples,
        instead of always favoring whichever output started with the
        highest random incoming weights (same fix as bandit's per-arm
        baseline / connectome_tictactoe_agent.py's calibrate_baseline)."""
        sums = {d: 0.0 for d in self.digits}
        n = 0
        for bitmap in calib_bitmaps:
            out = self.fly.step(image=bitmap)
            for d in self.digits:
                sums[d] += out[f"X_digit_{d}"]
            n += 1
        self.baseline = {d: sums[d] / max(n, 1) for d in self.digits}

    def predict(self, bitmap):
        if self.baseline is None:
            raise RuntimeError("call calibrate_baseline() before predict()")
        out = self.fly.step(image=bitmap)
        scores = {d: out[f"X_digit_{d}"] - self.baseline[d] for d in self.digits}
        pred = max(scores, key=scores.get)
        return pred, scores

    def train_on(self, bitmap, label, trials=3):
        self.fly.x_teach_output(f"digit_{label}", trials=trials, eta=self.teach_eta,
                                high=True, image=bitmap)
        others = [d for d in self.digits if d != label]
        wrong = int(self.rng.choice(others))
        self.fly.x_teach_output(f"digit_{wrong}", trials=1, eta=self.teach_eta,
                                high=False, image=bitmap)

    def drift_pct(self):
        """Same convention as connectome_tictactoe_agent.py's drift_pct()."""
        num = float(np.abs(self.fly.wM - self.fly.wM0).mean())
        den = float(np.abs(self.fly.wM0).mean()) + 1e-9
        return 100.0 * num / den

    def sleep(self, **kwargs):
        return self.fly.sleep(**kwargs)
