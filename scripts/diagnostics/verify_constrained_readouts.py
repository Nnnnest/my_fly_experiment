"""Sign-constrained readouts: non-negativity, optimality (KKT) and recovery of a planted solution on synthetic data.
    python -m scripts.diagnostics.verify_constrained_readouts"""
import numpy as np

from agents.shared.constrained_readouts import nnls_fista, nnls_fit_select

rng = np.random.default_rng(0)
n, d, lam = 300, 60, 1e-3
Z = np.hstack([np.abs(rng.standard_normal((n, d))), np.ones((n, 1))])
wt = np.maximum(rng.standard_normal(d), 0)
t = Z[:, :-1] @ wt - 1.0 + 0.05 * rng.standard_normal(n)
nn = np.ones(d + 1, bool)
nn[-1] = False
w = nnls_fista(Z, t, lam, nn, iters=5000)
g = Z.T @ (Z @ w - t) + lam * w
assert w[:-1].min() >= 0.0
assert np.abs(g[:-1][w[:-1] > 1e-8]).max() < 1e-2 and g[:-1][w[:-1] <= 1e-8].min() > -1e-2 and abs(g[-1]) < 1e-2
print("non-negativity and KKT conditions: OK; corr with planted weights:", round(float(np.corrcoef(w[:-1], wt)[0, 1]), 4))
y01 = (t > np.median(t)).astype(float)
w2, lam2 = nnls_fit_select(Z, y01, [1e-3, 1e-2, 1e-1], np.random.default_rng(0), nn)
print("fit_select: lambda", lam2, "train accuracy", round(float(((Z @ w2 > 0) == (y01 > 0.5)).mean()), 3))
print("OK")
