"""Tier 2 for counting: error-adaptive ("enforced") teaching.

Unlike grid-world/tic-tac-toe's Tier-2 gates (which SKIP training once a
value is already correct, because their failure mode was over-training /
non-decaying drift causing collapse -- see summary_addendum.txt section
6), counting's diagnostic points the other way: output_drift_pct() stays
small and the predicted-vs-actual table showed a real but heavily
under-scaled signal, not collapse. So this gate does the opposite: it
keeps calling x_teach_output on the SAME example, round-robin between a
few extra passes, for as long as the prediction remains off by more than
`tol`, up to `max_extra_trials` -- effort is spent where the current
error is largest, instead of a fixed `trials` count regardless of how
wrong the prediction already is.

Same fairness-discipline flag as prior tiers: no equivalent exists for
linear/mlp baselines here because their optimizers (SGD/Adam) already
have error-proportional step sizes built in -- an explicit extra-effort
gate on top would mostly duplicate a mechanism they already have, same
reasoning summary_addendum.txt gave for not building a qlearning/mlp
Tier-2 gate in tic-tac-toe.
"""
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_d = _HERE
for _ in range(6):
    if _d not in sys.path:
        sys.path.insert(0, _d)
    _d = os.path.dirname(_d)

import numpy as np

from agents.shapes.connectome_count_agent import ConnectomeCountAgent


class ConnectomeCountAgentV2(ConnectomeCountAgent):
    def __init__(self, n_max=4, n_out=8, hops=3, teach_eta=0.3, seed=0,
                 tol=0.5, max_extra_trials=6, trials_per_pass=2):
        super().__init__(n_max=n_max, n_out=n_out, hops=hops, teach_eta=teach_eta, seed=seed)
        self.tol = tol
        self.max_extra_trials = max_extra_trials
        self.trials_per_pass = trials_per_pass
        self.n_passes_used = []  # per-call record, for diagnostics

    def train_on(self, bitmap, count, trials=None):
        """Ignores the `trials` kwarg (kept only for interface parity with
        run_counting.py's call site) -- passes are decided by the error
        gate instead of a fixed count."""
        passes = 0
        for _ in range(1 + self.max_extra_trials):
            pred, _ = self.predict(bitmap)
            err = abs(pred - count)
            if err <= self.tol:
                break
            self.fly.x_teach_output("count", trials=self.trials_per_pass, eta=self.teach_eta,
                                    target=count / self.n_max, hops=self.hops, image=bitmap)
            passes += 1
        self.n_passes_used.append(passes)
        return passes

    def mean_passes(self, last_n=None):
        vals = self.n_passes_used[-last_n:] if last_n else self.n_passes_used
        return float(np.mean(vals)) if vals else 0.0
