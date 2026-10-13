import numpy as np


def calibrate_own_effect(agent, probe_triples):
    """Median value change from ONE reward update / ONE punish update (weights restored each time)."""
    w0 = agent.brain.wM.copy()
    dr, dp = [], []
    for a, b, c in probe_triples:
        v0 = agent.value(a, b, c)
        agent.update(a, b, c, True)
        v1 = agent.value(a, b, c)
        agent.brain.wM[:] = w0
        agent.cache.refresh()
        agent.update(a, b, c, False)
        v2 = agent.value(a, b, c)
        agent.brain.wM[:] = w0
        agent.cache.refresh()
        dr.append(v1 - v0)
        dp.append(v0 - v2)
    return max(float(np.median(dr)), 1e-9), max(float(np.median(dp)), 1e-9)


class GatedAdditionAgent:
    """Skip the update once the agent's own judgment already agrees with the label by a calibrated margin."""

    def __init__(self, inner, margin_frac=0.05):
        self.inner, self.margin_frac = inner, margin_frac
        self.tau_r = self.tau_p = 0.0
        self.n_trained = self.n_skipped = 0

    def calibrate_gate(self, probe_triples):
        d_r, d_p = calibrate_own_effect(self.inner, probe_triples)
        self.tau_r, self.tau_p = self.margin_frac * d_r, self.margin_frac * d_p

    def predict_many(self, triples):
        return self.inner.predict_many(triples)

    def value(self, a, b, c):
        return self.inner.value(a, b, c)

    def update(self, a, b, c, label):
        v = self.inner.value(a, b, c)
        if (v >= self.tau_r) if label else (v <= -self.tau_p):
            self.n_skipped += 1
            return
        self.n_trained += 1
        self.inner.update(a, b, c, label)

    def drift_pct(self):
        return self.inner.drift_pct()
