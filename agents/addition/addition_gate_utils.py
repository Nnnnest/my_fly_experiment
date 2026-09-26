"""Shared gate + calibration utilities for addition experiments, both
single-digit and two-digit -- works on any agent exposing .fly, .hops,
.encoder.encode(a,b,c), .value(a,b,c), .update(a,b,c,label) (both
ConnectomeAdditionAgent and ConnectomeAddition2Agent match this).

Same idea as grid-world/tic-tac-toe's error gate: skip train() once the
agent's own judgment on a triple is already correct beyond a margin, as an
external substitute for the decaying learning rate train() itself lacks.
Margin is calibrated in units of ONE train() call's actual effect on the
live circuit (calibrate_own_effect), not an arbitrary number.
"""
import numpy as np


def calibrate_own_effect(agent, probe_triples):
    """Median |value change| from one reward-train and one punish-train,
    each measured independently with weights restored between them.
    probe_triples: list of (a, b, c)."""
    w0 = agent.fly.wM.copy()
    dr, dp = [], []
    for a, b, c in probe_triples:
        v0 = agent.value(a, b, c)
        agent.fly.train(odor=agent.encoder.encode(a, b, c), reward=1.0, hops=agent.hops)
        v1 = agent.value(a, b, c)
        agent.fly.wM[:] = w0
        agent.fly.train(odor=agent.encoder.encode(a, b, c), punish=1.0, hops=agent.hops)
        v2 = agent.value(a, b, c)
        agent.fly.wM[:] = w0
        dr.append(v1 - v0)
        dp.append(v0 - v2)
    return max(float(np.median(dr)), 1e-9), max(float(np.median(dp)), 1e-9)


class GatedAdditionAgent:
    """Wraps a connectome addition agent; train() is skipped once the
    agent's own corrected judgment already agrees with the label by more
    than a calibrated margin. Only one gate mode here ('error') -- this
    task has 2 classes, not a per-state ranking over several actions, so
    there's no 'rank' variant to add."""

    def __init__(self, inner, margin_frac=0.05):
        self.inner = inner
        self.margin_frac = margin_frac
        self.tau_r = self.tau_p = 0.0
        self.n_trained = self.n_skipped = 0

    def calibrate_baseline(self, sample_triples):
        self.inner.calibrate_baseline(sample_triples)

    def calibrate_gate(self, probe_triples):
        d_r, d_p = calibrate_own_effect(self.inner, probe_triples)
        self.tau_r = self.margin_frac * d_r
        self.tau_p = self.margin_frac * d_p
        print(f"[gate] one-train effect reward={d_r:.3f} punish={d_p:.3f}; "
              f"margins {self.tau_r:.3f}/{self.tau_p:.3f} (margin_frac={self.margin_frac})")

    def predict(self, a, b, c):
        return self.inner.predict(a, b, c)

    def value(self, a, b, c):
        return self.inner.value(a, b, c)

    def update(self, a, b, c, label):
        v = self.inner.value(a, b, c)
        satisfied = (v >= self.tau_r) if label else (v <= -self.tau_p)
        if satisfied:
            self.n_skipped += 1
            return
        self.n_trained += 1
        self.inner.update(a, b, c, label)

    def drift_pct(self):
        return self.inner.drift_pct()

    def sleep(self, **kwargs):
        return self.inner.sleep(**kwargs)
