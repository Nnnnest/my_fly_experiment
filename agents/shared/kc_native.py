"""The native mb-mode plasticity rule of FlyBrainAPI.train(), re-expressed on fixed KC codes.

Everything here is derived from reading fly_api.py (mb mode, hops=1, rate mode, no sleep, assoc=0, no x_remember codes):

  1. KC activity kcs = ReLU(sum_w ALPN->KC drive / fan) (plus the DAN drive of the US pathway).
  2. Plasticity mask m = kcs >= (k-th largest kcs), k = int(0.05 * n_kc).
     If fewer than k KCs are positive, the k-th largest value is 0 and m selects EVERY KC, also silent ones.
     The readout (step(), MBValueCache) uses kcs * m, so silent KCs add nothing there: plasticity mask and
     readout code then differ.
  3. Third factor s = clip(DAN response / DAN reference, 0, 2).
  4. reward: KC->avoid-MBON edges of masked KCs are multiplied by (1 - 0.15 s);
     punish: KC->"approach" edges (every non-avoid MBON) likewise. Floor 0.05.
  5. Per-MBON renormalisation: scale = clip(ref_mb / current_total, 0.9, 1.1).

apply_rule() with default arguments reproduces train() bit for bit (checked by
scripts/diagnostics/verify_kc_native.py). Its keyword arguments relax ONE native property each.
"""
import numpy as np

KC_FRAC = 0.05
ETA = 0.15
W_MIN = 0.05


def kc_k(n_kc, kc_frac=KC_FRAC):
    return max(1, int(n_kc * kc_frac))


def native_mask(kcs, kc_frac=KC_FRAC):
    """Plasticity mask exactly as in FlyBrainAPI.train()."""
    kcs = np.asarray(kcs)
    return kcs >= np.sort(kcs)[-kc_k(len(kcs), kc_frac)]


def mask_stats(kcs, kc_frac=KC_FRAC):
    """kcs: (n_kc,) or (n, n_kc). Returns arrays over rows:
    n_pos (KCs with positive activity), n_plast (KCs in the plasticity mask), n_read (positive KCs in the mask =
    size of the readout code), plus ints k and n_kc."""
    K = np.atleast_2d(np.asarray(kcs))
    k = kc_k(K.shape[1], kc_frac)
    thr = np.sort(K, axis=1)[:, -k]
    m = K >= thr[:, None]
    pos = K > 0
    return {"n_pos": pos.sum(1), "n_plast": m.sum(1), "n_read": (pos & m).sum(1), "k": k, "n_kc": K.shape[1]}


def kc_activity(cache, X):
    """KC activity before the top-k step, from an MBValueCache (same formula as cache._codes). X: (n, n_alpn)."""
    X = np.atleast_2d(np.asarray(X, np.float32))
    n = min(X.shape[1], cache.n_alpn)
    drive = (X[:, :n] * np.float32(2.0)) @ cache.M[:n]
    return np.maximum(0.0, drive / cache.fan_kc).astype(np.float32)


def dan_reference(brain, hops=1, thr=0.0):
    """US-alone DAN response used to calibrate the third factor (same code and cache attribute as train())."""
    ref = getattr(brain, "_dan_ref", None)
    if ref is None:
        idx, val, _ = brain.encode(dan_rew=1.0, dan_pun=1.0)
        h = brain._forward_pure_mb(idx, val, hops=hops, thr=thr)
        ref = {"pam": float(h[np.asarray(brain.dan_pam, dtype=np.int32)].mean()) if len(brain.dan_pam) else 0.0,
               "pun": float(h[np.asarray(brain.dan_ppl, dtype=np.int32)].mean()) if len(brain.dan_ppl) else 0.0}
        brain._dan_ref = ref
    return ref


def train_inputs(brain, vec, label, hops=1, thr=0.0):
    """What train(odor=vec, reward=1.0 / punish=1.0) computes before it touches any weight.
    Returns (kcs, m, s). Runs the simulator's own forward pass, including the US drive of the DAN neurons."""
    rw, pu = (1.0, 0.0) if label else (0.0, 1.0)
    idx, val, _ = brain.encode(odor=vec, dan_rew=rw, dan_pun=pu)
    h = brain._forward_pure_mb(idx, val, hops=hops, thr=thr)
    kcs = h[brain.KC]
    m = native_mask(kcs)
    ref = dan_reference(brain, hops, thr)
    if label:
        d = float(h[np.asarray(brain.dan_pam, dtype=np.int32)].mean()) if len(brain.dan_pam) else 0.0
        s = float(np.clip(d / (ref["pam"] + 1e-9), 0.0, 2.0))
    else:
        d = float(h[np.asarray(brain.dan_ppl, dtype=np.int32)].mean()) if len(brain.dan_ppl) else 0.0
        s = float(np.clip(d / (ref["pun"] + 1e-9), 0.0, 2.0))
    return kcs, m, s


def apply_rule(brain, kcs, m, s, label, renorm=True, push_pull=False, decay_n0=None, counts=None,
               mask_active=False):
    """Apply one update to brain.wM (KC->MBON edges only), like train() does.
    Defaults = the native rule. Each keyword relaxes one native property:
      renorm=False     no per-MBON renormalisation
      push_pull=True   the opposite MBON population is potentiated by the same step (native: depression only)
      decay_n0=N0      step / (1 + n_k / N0), n_k = times KC k was in the mask so far (native: fixed step);
                       needs counts, a float array of length n_kc that is updated in place
      mask_active=True plasticity mask restricted to KCs with positive activity (native: also silent KCs
                       when fewer than k KCs are active)
    Returns the number of selected (depressed) edges."""
    km = brain.km_mask
    w = brain.wM[km]
    mm = (m & (np.asarray(kcs) > 0)) if mask_active else m
    act = mm[brain.km_ki]
    sel = (brain.km_is_avoid if label else brain.km_is_approach) & act
    if not push_pull and decay_n0 is None:
        w = np.where(sel, w * (1.0 - ETA * s * 1.0), w)          # exactly train()'s expression
    else:
        step = ETA * s
        if decay_n0 is not None:
            if counts is None:
                raise ValueError("decay_n0 needs counts")
            step = step / (1.0 + counts[brain.km_ki] / float(decay_n0))
        w = np.where(sel, w * (1.0 - step), w)
        if push_pull:
            pot = (brain.km_is_approach if label else brain.km_is_avoid) & act
            w = np.where(pot, w * (1.0 + step), w)
        if decay_n0 is not None:
            counts += mm
    w = np.maximum(w, W_MIN)
    if renorm:
        cur = np.zeros(len(brain.MBON))
        np.add.at(cur, brain.km_mi, w.astype(float))
        sc = np.ones(len(brain.MBON))
        nz = cur > 1e-9
        sc[nz] = np.clip(brain.ref_mb[nz] / cur[nz], 0.9, 1.1)
        w = w * sc[brain.km_mi]
    brain.wM[km] = np.asarray(w).astype(np.float32)
    return int(sel.sum())
