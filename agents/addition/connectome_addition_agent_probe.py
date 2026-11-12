import numpy as np

from agents.addition.connectome_addition_agent import ConnectomeAdditionAgent
from agents.shared.kc_native import train_inputs, mask_stats, kc_activity, apply_rule


class ProbeAdditionAgent(ConnectomeAdditionAgent):
    """ConnectomeAdditionAgent with (a) optional per-update instrumentation of the plasticity mask,
    (b) optional skipping of degenerate updates, (c) an optional replacement of train() by apply_rule().
    rule=None  : the native train() (identical to ConnectomeAdditionAgent)
    rule={...} : apply_rule(**rule) on the same KC codes; {} is the native rule re-implemented
    skip=None | 'zero' (no KC active) | 'mismatch' (plasticity mask larger than the readout code)
    Baseline calibration, prediction and evaluation are inherited unchanged."""

    def __init__(self, encoder, width, rule=None, skip=None, record=False):
        super().__init__(encoder, width)
        if skip not in (None, "zero", "mismatch"):
            raise ValueError("skip must be None, 'zero' or 'mismatch'")
        self.rule, self.skip, self.record = rule, skip, record
        self.counts = np.zeros(len(self.brain.KC))
        self.rows = []
        self.n_trained = self.n_skipped = self.n_cache_disagree = 0

    def update(self, a, b, c, label):
        label = bool(label)
        b_ = self.brain
        vec = self.encoder.encode(a, b, c)
        kcs, m, s = train_inputs(b_, vec, label)
        st = mask_stats(kcs)
        n_pos, n_plast, n_read = int(st["n_pos"][0]), int(st["n_plast"][0]), int(st["n_read"][0])
        skipped = (self.skip == "zero" and n_pos == 0) or (self.skip == "mismatch" and n_plast > n_read)
        if self.record:
            self.n_cache_disagree += int(not np.allclose(kc_activity(self.cache, vec)[0], kcs, rtol=1e-4, atol=1e-5))
            n_sel = int(((b_.km_is_avoid if label else b_.km_is_approach) & m[b_.km_ki]).sum())
            w0 = b_.wM[b_.km_mask].copy()
        if skipped:
            self.n_skipped += 1
        else:
            self.n_trained += 1
            if self.rule is None:
                super().update(a, b, c, label)
            else:
                apply_rule(b_, kcs, m, s, label, counts=self.counts, **self.rule)
                self.cache.refresh()
        if self.record:
            n_changed, rel = 0, 0.0
            if not skipped:
                r = np.abs(b_.wM[b_.km_mask] - w0) / np.maximum(w0, 1e-9)
                n_changed, rel = int((r > 1e-3).sum()), float(r.mean())
            self.rows.append({"label": float(label), "n_pos": n_pos, "n_plast": n_plast, "n_read": n_read,
                              "n_sel": n_sel, "skipped": float(skipped), "n_changed": n_changed, "rel_change": rel})
