"""Exact fast value evaluator for FlyBrainAPI(mode="mb") at hops=1.

Why this is exact, not an approximation (from reading fly_api.py):
  * In mb mode train() only modifies KC->MBON weights (km_mask edges). The
    ALPN->KC wiring is frozen, so the top-5% KC code of every (state, action)
    pattern is a constant at hops=1 (one hop: stimulus clamps ALPN, KC =
    relu(sum w*a / fan)).
  * MB_pref = mean(approach MBON) - mean(avoid MBON) is linear in the KC->MBON
    weights, so value(pair) = code(pair) . Wc with Wc a per-KC scalar that is
    rebuilt from the current weights with one bincount after each train().

So select_action needs zero brain.step() calls. The class mirrors the library's
indexing (including km_ki vs brain.KC order) and can verify itself against
brain.step() at any time, so a mismatch is detected rather than trusted.
"""
import numpy as np


class MBValueCache:
    def __init__(self, brain, encoder, kc_frac=0.05):
        if brain.mode != "mb":
            raise ValueError("MBValueCache supports mode='mb' only")
        self._check_static(brain)
        if getattr(brain, "_fan", None) is None:
            brain.enable_scaling()
        b = brain
        self.brain, self.encoder, self.kc_frac = b, encoder, kc_frac
        alpn, kc = np.asarray(b.ALPN), np.asarray(b.KC)
        self.n_alpn, self.n_kc = len(alpn), len(kc)

        a_idx = np.full(b.N, -1, np.int64); a_idx[alpn] = np.arange(len(alpn))
        k_idx = np.full(b.N, -1, np.int64); k_idx[kc] = np.arange(len(kc))
        sw = (b.wM * b.sign).astype(np.float32)
        m = (a_idx[b.pre] >= 0) & (k_idx[b.post] >= 0)
        M = np.zeros((self.n_alpn, self.n_kc), np.float32)
        np.add.at(M, (a_idx[b.pre[m]], k_idx[b.post[m]]), sw[m])
        self.M = M
        self.fan_kc = (b._fan[kc] + 1e-6).astype(np.float32)

        pos = b._mb_pos
        ai = [pos[int(a)] for a in np.sort(b.approach)]
        vi = [pos[int(a)] for a in np.sort(b.avoid)]
        c = np.zeros(len(b.MBON))
        np.add.at(c, ai, 1.0 / len(ai))
        np.add.at(c, vi, -1.0 / len(vi))
        self.c = c

        self.codes = self._codes(encoder.patterns)
        self.n_actions = encoder.n_actions
        self.kc_sorted = bool(np.all(np.diff(kc) > 0))
        self.refresh()

    @staticmethod
    def _check_static(b):
        bad = []
        if getattr(b, "_act", "relu") != "relu": bad.append("activation != relu")
        if getattr(b, "_scaling", "static") != "static": bad.append("scaling != static")
        if float(getattr(b, "_state_leak", 0.0) or 0.0) != 0.0: bad.append("state leak on")
        std = getattr(b, "_std", None)
        if std is not None and std["alpha"] > 0: bad.append("STD on")
        if getattr(b, "_auto_sleep", None) is not None: bad.append("auto-sleep on")
        if getattr(b, "_clock", None) is not None: bad.append("clock on")
        if bad:
            raise ValueError("cache needs default pure/static settings; found: " + ", ".join(bad))

    def _codes(self, pat):
        n = min(pat.shape[1], self.n_alpn)
        drive = (np.asarray(pat[:, :n], np.float32) * np.float32(2.0)) @ self.M[:n]
        kcs = np.maximum(0.0, drive / self.fan_kc).astype(np.float32)
        k = max(1, int(self.n_kc * self.kc_frac))
        thr = np.sort(kcs, axis=1)[:, -k][:, None]
        return (kcs * (kcs >= thr)).astype(np.float32)

    def refresh(self):
        """Rebuild the per-KC valence weight from current KC->MBON weights."""
        b = self.brain
        w = b.wM[b.km_mask].astype(np.float64)
        self.Wc = np.bincount(b.km_ki, weights=w * self.c[b.km_mi], minlength=self.n_kc)

    def values_pairs(self, pair_ids):
        return self.codes[pair_ids] @ self.Wc

    def values_state(self, state):
        ids = state * self.n_actions + np.arange(self.n_actions)
        return self.values_pairs(ids)

    def values_all(self):
        return self.codes @ self.Wc

    def scan_live(self, mag=5.0):
        """ALPN positions whose one-hot drive has a measurable MB_pref effect at
        hops=1 (analytic replacement for the brain.step() scan behind
        live_alpn_indices.npy; compare the count with the saved file)."""
        pat = np.eye(self.n_alpn, dtype=np.float32) * mag
        vals = self._codes(pat) @ self.Wc
        return np.where(np.abs(vals) > 1e-9)[0]

    def overlap_stats(self):
        """Interference diagnostic: Jaccard overlap between pairs' KC codes."""
        B = (self.codes > 0).astype(np.float32)
        inter = B @ B.T
        sz = B.sum(1)
        union = sz[:, None] + sz[None, :] - inter
        jac = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        iu = np.triu_indices(len(B), 1)
        j = jac[iu]
        return {"pairs": int(len(B)), "mean_kc_active": float(sz.mean()),
                "empty_codes": int((sz == 0).sum()),
                "jaccard_mean": float(j.mean()), "jaccard_p95": float(np.quantile(j, 0.95)),
                "frac_pairs_jaccard_gt_0.2": float((j > 0.2).mean())}

    def verify(self, n=16, seed=0):
        """Compare cache to brain.step() MB_pref on n random pairs."""
        self.refresh()
        rng = np.random.default_rng(seed)
        n_pairs = len(self.codes)
        ids = rng.choice(n_pairs, size=min(n, n_pairs), replace=False)
        ref = np.array([self.brain.step(odor=self.encoder.patterns[i], hops=1)["MB_pref"]
                        for i in ids])
        got = self.values_pairs(ids)
        err = float(np.abs(ref - got).max())
        scale = float(np.abs(ref).max())
        return {"max_abs_err": err, "scale": scale,
                "ok": err <= 1e-3 * max(scale, 1e-3), "kc_sorted": self.kc_sorted}


def train_no_readout(brain, odor, hops=1, **kw):
    """brain.train() without its trailing brain.step() readout (weights identical,
    one forward pass cheaper). Verified bit-identical on wM."""
    brain.step = lambda *a, **k: {}
    try:
        brain.train(odor=odor, hops=hops, **kw)
    finally:
        del brain.step


def ensure_live_hops1(path, brain=None):
    """Load live ALPN positions from `path`; if missing, compute them analytically
    from the circuit (MBValueCache.scan_live) and save. Needs fly_api on sys.path."""
    import os
    if os.path.exists(path):
        return np.load(path)
    from fly_api import FlyBrainAPI
    from sparse_encoder import SparseStateActionEncoder
    brain = brain or FlyBrainAPI(mode="mb")
    dummy = SparseStateActionEncoder(2, 2, np.arange(20), k=2, max_overlap=2)
    live = MBValueCache(brain, dummy).scan_live()
    np.save(path, live)
    print(f"[live] {path} missing -> computed {len(live)} live ALPN positions and saved it")
    return live


def calibrate_own_effect(brain, cache, encoder, n_probe=8, seed=12345):
    """Median change in a pair's own value caused by ONE reward train and by ONE
    punish train (magnitudes, > 0). Weights are restored exactly afterwards.
    Used to set error-gate margins in units the circuit actually produces."""
    w0 = brain.wM.copy()
    rng = np.random.default_rng(seed)
    n = len(encoder.patterns)
    dr, dp = [], []
    for pid in rng.choice(n, size=min(n_probe, n), replace=False):
        odor = encoder.patterns[pid]
        cache.refresh(); v0 = cache.values_pairs([pid])[0]
        train_no_readout(brain, odor, hops=1, reward=1.0)
        cache.refresh(); v1 = cache.values_pairs([pid])[0]
        brain.wM[:] = w0
        train_no_readout(brain, odor, hops=1, punish=1.0)
        cache.refresh(); v2 = cache.values_pairs([pid])[0]
        brain.wM[:] = w0
        dr.append(v1 - v0); dp.append(v0 - v2)
    cache.refresh()
    return max(float(np.median(dr)), 1e-9), max(float(np.median(dp)), 1e-9)
