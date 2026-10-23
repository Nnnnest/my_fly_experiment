"""Linear readouts on fixed feature vectors (numpy only): online delta rule and offline ridge.
Used by scripts/readouts/run_readouts.py to read the SAME KC codes the native plasticity rule sees."""
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def add_bias(X):
    return np.hstack([X, np.ones((len(X), 1))])


def prep_l2(X):
    """Row-wise L2 normalisation (+ bias column). Used for KC codes."""
    X = np.asarray(X, np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return add_bias(X / np.maximum(n, 1e-12))


def prep_scaled(X_train, *others):
    """Divide by the training maximum (+ bias column). Used for raw input vectors. Returns [train, *others]."""
    X_train = np.asarray(X_train, np.float64)
    s = max(float(np.abs(X_train).max()), 1e-12)
    return [add_bias(X_train / s)] + [add_bias(np.asarray(o, np.float64) / s) for o in others]


def accuracy(X, y01, w):
    return float(((X @ w > 0) == (np.asarray(y01) > 0.5)).mean())


def delta_fit(X, y01, alpha):
    """Online logistic delta rule, one pass in the given order (same order as the native agent's stream)."""
    w = np.zeros(X.shape[1])
    for x, y in zip(X, y01):
        w += alpha * (y - sigmoid(w @ x)) * x
    return w


def ridge_path(X, t, lams):
    """Ridge solutions for several lambdas from one eigendecomposition. t: targets (+1/-1)."""
    X, t = np.asarray(X, np.float64), np.asarray(t, np.float64)
    n, d = X.shape
    if n <= d:                                   # dual: w = X^T (XX^T + lam I)^-1 t
        e, V = np.linalg.eigh(X @ X.T)
        Vt = V.T @ t
        return [X.T @ (V @ (Vt / (np.maximum(e, 0.0) + lam))) for lam in lams]
    e, V = np.linalg.eigh(X.T @ X)               # primal: w = (X^T X + lam I)^-1 X^T t
    Vb = V.T @ (X.T @ t)
    return [V @ (Vb / (np.maximum(e, 0.0) + lam)) for lam in lams]


def ridge_fit_select(X, y01, lams, rng, val_frac=0.2):
    """Choose lambda on a held-out part of the TRAINING set only, then refit on all training data."""
    y01 = np.asarray(y01, float)
    t = 2.0 * y01 - 1.0
    idx = rng.permutation(len(X))
    k = int((1.0 - val_frac) * len(X))
    tr, va = idx[:k], idx[k:]
    ws = ridge_path(X[tr], t[tr], lams)
    scores = [accuracy(X[va], y01[va], w) for w in ws]
    lam = lams[int(np.argmax(scores))]
    return ridge_path(X, t, [lam])[0], lam
