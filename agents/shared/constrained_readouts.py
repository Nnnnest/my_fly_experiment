"""Offline readouts on fixed KC codes with sign constraints (numpy only).
Model: value = Z w, w_j >= 0 for the coordinates flagged in `nonneg`, free elsewhere (the bias column).
Fit by FISTA on 0.5||Zw - t||^2 + 0.5 lam ||w||^2 (fixed iteration count; the penalty is chosen on a hold-out part of
the TRAINING data only, as in readouts.ridge_fit_select)."""
import numpy as np

from agents.shared.readouts import accuracy


def _lipschitz(Z, iters=30):
    v = np.random.default_rng(0).standard_normal(Z.shape[1])
    v /= np.linalg.norm(v)
    for _ in range(iters):
        u = Z.T @ (Z @ v)
        n = np.linalg.norm(u)
        if n == 0.0:
            return 1.0
        v = u / n
    return float(v @ (Z.T @ (Z @ v)))


def nnls_fista(Z, t, lam, nonneg, iters=300):
    Z, t = np.asarray(Z, np.float64), np.asarray(t, np.float64)
    L = 1.1 * _lipschitz(Z) + lam
    w = np.zeros(Z.shape[1])
    y, tk = w.copy(), 1.0
    for _ in range(iters):
        g = Z.T @ (Z @ y - t) + lam * y
        wn = y - g / L
        wn[nonneg] = np.maximum(wn[nonneg], 0.0)
        tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * tk * tk))
        y = wn + ((tk - 1.0) / tn) * (wn - w)
        w, tk = wn, tn
    return w


def nnls_fit_select(Z, y01, lams, rng, nonneg, val_frac=0.2, iters=300):
    y01 = np.asarray(y01, float)
    t = 2.0 * y01 - 1.0
    idx = rng.permutation(len(Z))
    k = int((1.0 - val_frac) * len(Z))
    tr, va = idx[:k], idx[k:]
    scores = [accuracy(Z[va], y01[va], nnls_fista(Z[tr], t[tr], lam, nonneg, iters)) for lam in lams]
    lam = lams[int(np.argmax(scores))]
    return nnls_fista(Z, t, lam, nonneg, iters), lam


def nonneg_design(K):
    """K: KC codes with a trailing bias column (readouts.prep_l2). One non-negative weight per KC, free bias."""
    nn = np.ones(K.shape[1], bool)
    nn[-1] = False
    return K, nn


def anatomy_design(K, has_app, has_avo):
    """Two non-negative channels per KC like the circuit: value = sum_k x_k (w_app_k - w_avo_k) + bias, where a KC can
    only feed a channel if it has at least one KC->MBON edge to that population (has_app / has_avo, bool over KCs)."""
    X = K[:, :-1]
    Z = np.hstack([X * has_app[None, :], -(X * has_avo[None, :]), K[:, -1:]])
    nn = np.ones(Z.shape[1], bool)
    nn[-1] = False
    return Z, nn


def edge_flags(brain, cache):
    """has_app / has_avo over KCs (columns of the KC code), from the real KC->MBON edges and the readout vector c."""
    n_kc = len(brain.KC)
    c = cache.c[brain.km_mi]
    return (np.bincount(brain.km_ki[c > 0], minlength=n_kc) > 0,
            np.bincount(brain.km_ki[c < 0], minlength=n_kc) > 0)
