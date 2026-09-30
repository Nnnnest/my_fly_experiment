"""KC-code geometry diagnostic: are the REAL mushroom-body codes worse than random ones?

Question this script answers on the REAL circuit (mb_circuit.npz):
  In the 31x21 grid-world runs, kc_delta (real KC code + external delta rule) did worse than
  rand_delta (random binary code, same size/sparsity). WHY? This script measures it directly,
  without running any maze, by comparing four families of per-state codes:

    REAL          state -> sparse k-hot ALPN pattern -> real ALPN->KC wiring -> top-kc_frac KCs
    RAND_UNIFORM  random KC subsets, same size per state (the original 'rand_delta' control)
    RAND_USAGE    random KC subsets, same size per state AND the same per-KC popularity as REAL
                  (separates "some KCs are hubs" from "the wiring correlates states")
    WIRING_SHUF   same ALPN patterns, but ALPN->KC wiring shuffled inside every KC column
                  (keeps each KC's input strengths, destroys which ALPN feeds which KC)

For every family it reports:
  Geometry   : mean / 99th-percentile cosine between different states' codes, eigenvalues of the
               normalised similarity matrix (eig_min small = some state directions learn slowly),
               effective rank, fraction of 'slow' directions (eig < 0.1), KC usage concentration.
  Learning   : normalised-LMS (delta rule) fit of random +-1 targets, MSE after 10/50/E epochs.

Run (from anywhere inside my_experiments, with the fly-brain files reachable):
  python scripts/diagnostics/kc_geometry_diagnostic.py
  python scripts/diagnostics/kc_geometry_diagnostic.py --mazes 19x13 25x17 31x21 --seeds 5
Output: printed tables + results/gridworld/kc_geometry_diagnostic.csv
"""
import os
import sys
import csv
import argparse
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_up(start, marker, levels=6):
    d = start
    for _ in range(levels):
        if os.path.exists(os.path.join(d, marker)):
            return d
        d = os.path.dirname(d)
    return None


# ----------------------------------------------------------------------------------------------
# core maths (no dependency on the fly circuit -> testable on synthetic data)
# ----------------------------------------------------------------------------------------------
def codes_from_M(patterns, M, fan_kc, kc_frac):
    """Exactly MBValueCache._codes, but with an explicit wiring matrix M (so it can be shuffled)."""
    n_alpn, n_kc = M.shape
    n = min(patterns.shape[1], n_alpn)
    drive = (np.asarray(patterns[:, :n], np.float32) * np.float32(2.0)) @ M[:n]
    kcs = np.maximum(0.0, drive / fan_kc).astype(np.float32)
    k = max(1, int(n_kc * kc_frac))
    thr = np.sort(kcs, axis=1)[:, -k][:, None]
    return (kcs * (kcs >= thr)).astype(np.float32)


def binarize(codes):
    return (codes > 0)


def random_uniform(counts, n_kc, rng):
    B = np.zeros((len(counts), n_kc), bool)
    for i, c in enumerate(counts):
        if c > 0:
            B[i, rng.choice(n_kc, size=int(c), replace=False)] = True
    return B


def random_matched_usage(counts, usage, rng):
    """Random codes with the same size per state and the same per-KC popularity as `usage`."""
    n_kc = len(usage)
    p = usage.astype(np.float64) + 1e-6 * max(usage.mean(), 1e-9)
    p /= p.sum()
    B = np.zeros((len(counts), n_kc), bool)
    for i, c in enumerate(counts):
        if c > 0:
            B[i, rng.choice(n_kc, size=int(c), replace=False, p=p)] = True
    return B


def geometry(B):
    """Similarity / conditioning metrics for a boolean (n_states x n_kc) code matrix."""
    B = B.astype(np.float64)
    n = len(B)
    norms = np.linalg.norm(B, axis=1)
    norms[norms == 0] = 1.0
    Bn = B / norms[:, None]
    C = Bn @ Bn.T                                   # cosine similarity, diag = 1
    off = C[np.triu_indices(n, 1)]
    ev = np.clip(np.linalg.eigvalsh(C), 0.0, None)  # mean eigenvalue = 1 (trace = n)
    usage = B.sum(0)
    srt = np.sort(usage)[::-1]
    top1 = srt[:max(1, int(0.01 * len(srt)))].sum() / max(usage.sum(), 1.0)
    return {
        "cos_mean": float(off.mean()),
        "cos_p99": float(np.quantile(off, 0.99)),
        "eig_min": float(ev[0]),
        "eig_max": float(ev[-1]),
        "eff_rank_frac": float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-12) / n),
        "slow_frac": float((ev < 0.1).mean()),
        "kc_used_frac": float((usage > 0).mean()),
        "top1pct_kc_share": float(top1),
        "empty_codes": int((B.sum(1) == 0).sum()),
    }


def nlms_curve(B, y, alpha, epochs, rng):
    """Normalised delta rule on binary codes: w[active] += alpha * err / n_active.
    Returns the MSE over all states after each epoch."""
    n, n_kc = B.shape
    idx = [np.flatnonzero(B[i]) for i in range(n)]
    Bf = B.astype(np.float64)
    w = np.zeros(n_kc)
    out = []
    for _ in range(epochs):
        for s in rng.permutation(n):
            ii = idx[s]
            if len(ii) == 0:
                continue
            err = y[s] - w[ii].sum()
            w[ii] += alpha * err / len(ii)
        out.append(float(np.mean((y - Bf @ w) ** 2)))
    return np.array(out)


def evaluate_families(patterns, M, fan_kc, kc_frac, seed, alpha, epochs):
    """Build the four code families for one encoder seed and measure each."""
    rng = np.random.default_rng(10_000 + seed)
    n_states = len(patterns)
    n_kc = M.shape[1]

    B_real = binarize(codes_from_M(patterns, M, fan_kc, kc_frac))
    counts = B_real.sum(1)
    usage = B_real.sum(0)
    B_uni = random_uniform(counts, n_kc, rng)
    B_use = random_matched_usage(counts, usage, rng)
    M_shuf = rng.permuted(M, axis=0)                # shuffle rows independently inside each KC column
    B_shuf = binarize(codes_from_M(patterns, M_shuf, fan_kc, kc_frac))

    y = rng.choice([-1.0, 1.0], size=n_states)      # same targets for every family in this seed
    res = {}
    for name, B in (("REAL", B_real), ("RAND_UNIFORM", B_uni),
                    ("RAND_USAGE", B_use), ("WIRING_SHUF", B_shuf)):
        g = geometry(B)
        curve = nlms_curve(B, y, alpha, epochs, np.random.default_rng(seed))
        g["mse_ep10"] = float(curve[min(9, epochs - 1)])
        g["mse_ep50"] = float(curve[min(49, epochs - 1)])
        g["mse_final"] = float(curve[-1])
        g["mean_active"] = float(B.sum(1).mean())
        res[name] = g
    return res


METRICS = ["cos_mean", "cos_p99", "eig_min", "eff_rank_frac", "slow_frac",
           "kc_used_frac", "top1pct_kc_share", "mse_ep10", "mse_ep50", "mse_final"]


def summarise(rows):
    """rows: list of dict(family=..., metric values) -> {family: {metric: (mean, std)}}"""
    out = {}
    for fam in ("REAL", "RAND_UNIFORM", "RAND_USAGE", "WIRING_SHUF"):
        sub = [r for r in rows if r["family"] == fam]
        out[fam] = {m: (float(np.mean([r[m] for r in sub])), float(np.std([r[m] for r in sub])))
                    for m in METRICS}
    return out


def print_table(title, summ):
    print(f"\n{title}")
    hdr = f"{'family':<13}" + "".join(f"{m:>17}" for m in METRICS)
    print(hdr)
    print("-" * len(hdr))
    for fam, d in summ.items():
        print(f"{fam:<13}" + "".join(f"{d[m][0]:>10.4f}±{d[m][1]:<5.3f}" for m in METRICS))


HOW_TO_READ = """
HOW TO READ (compare rows within one block; higher cos / lower eig_min / higher mse = harder to learn)
  1. REAL vs RAND_UNIFORM: this is the original kc_delta vs rand_delta gap. If REAL has clearly higher
     cos_p99 / slow_frac / mse_* than RAND_UNIFORM, the real codes really are harder to learn on.
     If they are about the same, the code geometry does NOT explain the maze result.
  2. REAL vs RAND_USAGE: same per-KC popularity as REAL, but random otherwise.
     If RAND_USAGE ~ REAL (both worse than RAND_UNIFORM) -> the cause is uneven KC usage
     (some KCs fire for many states, many KCs almost never) -- see kc_used_frac / top1pct_kc_share.
     If REAL is still worse than RAND_USAGE -> the specific wiring adds correlation on top of that.
  3. WIRING_SHUF: same inputs, shuffled ALPN->KC connections. If it looks like RAND_*, the specific
     wiring matters; if it looks like REAL, the effect comes from KC input-strength statistics.
  4. Check the trend across mazes/sizes: a real 'less independent at scale' effect should GROW with n_states.
  5. Numbers are mean±std over seeds (each seed = different state->ALPN assignment).
Nothing here is proof about the maze results by itself; it tells you which explanation the real
circuit supports. If the differences are within the ± spread, say so in the article.
"""


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mazes", nargs="+", default=["19x13", "25x17", "31x21"],
                    help="maze sizes WxH (odd numbers); n_states = traversable cells, as in the runs")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--k", type=int, default=6, help="ALPNs per state code (as in the runs)")
    ap.add_argument("--max-overlap", type=int, default=1)
    ap.add_argument("--kc-fracs", type=float, nargs="+", default=[0.05, 0.02, 0.01])
    ap.add_argument("--alpha", type=float, default=0.5, help="normalised LMS step")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--exp-root", default=None, help="my_experiments dir (auto-detected)")
    ap.add_argument("--fly-root", default=None, help="dir with fly_api.py + mb_circuit.npz (auto-detected)")
    ap.add_argument("--live", default=None, help="live_alpn_indices.npy (default results/gridworld/)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    exp_root = a.exp_root or _find_up(HERE, "envs")
    fly_root = a.fly_root or _find_up(HERE, "fly_api.py")
    if not exp_root or not fly_root:
        raise SystemExit(f"cannot locate roots (exp_root={exp_root}, fly_root={fly_root}); pass --exp-root/--fly-root")
    for p in (exp_root, os.path.join(exp_root, "agents", "shared"), fly_root):
        if p not in sys.path:
            sys.path.insert(0, p)

    from fly_api import FlyBrainAPI
    from mb_value_cache import MBValueCache, ensure_live_hops1
    from sparse_encoder import SparseStateActionEncoder
    try:
        from envs.maze_generator import generate_maze, count_traversable
    except ImportError:
        from maze_generator import generate_maze, count_traversable

    live_path = a.live or os.path.join(exp_root, "results", "gridworld", "live_alpn_indices.npy")
    out_path = a.out or os.path.join(exp_root, "results", "gridworld", "kc_geometry_diagnostic.csv")

    print("== loading the real circuit ==")
    brain = FlyBrainAPI(mode="mb", path=fly_root)
    live = ensure_live_hops1(live_path, brain)
    print(f"live ALPN positions: {len(live)} (expected 302 at hops=1)")

    # --- sanity checks: our code path must equal the library's ------------------------------
    chk_enc = SparseStateActionEncoder(16, 1, live, k=a.k, max_overlap=a.max_overlap, seed=123)
    cache = MBValueCache(brain, chk_enc, kc_frac=0.05)
    ver = cache.verify(n=16)
    print(f"[check 1] cache vs brain.step(): {ver}")
    mine = codes_from_M(chk_enc.patterns, cache.M, cache.fan_kc, 0.05)
    same = bool(np.allclose(mine, cache.codes))
    print(f"[check 2] my codes_from_M == MBValueCache.codes: {same}")
    if not (ver["ok"] and same):
        raise SystemExit("sanity checks failed -> do not trust anything below")
    M, fan_kc = cache.M, cache.fan_kc
    print(f"n_alpn={M.shape[0]}  n_kc={M.shape[1]}")

    all_rows = []
    t0 = time.time()
    for spec in a.mazes:
        w, h = (int(x) for x in spec.lower().split("x"))
        n_states = count_traversable(generate_maze(w, h, extra_connections=0.1, seed=0))
        for kc_frac in a.kc_fracs:
            rows = []
            for seed in range(a.seeds):
                try:
                    enc = SparseStateActionEncoder(n_states, 1, live, k=a.k,
                                                   max_overlap=a.max_overlap, seed=seed)
                except RuntimeError as e:
                    print(f"  maze {spec} ({n_states} states) skipped: {e}")
                    rows = []
                    break
                res = evaluate_families(enc.patterns, M, fan_kc, kc_frac, seed, a.alpha, a.epochs)
                for fam, g in res.items():
                    r = {"maze": spec, "n_states": n_states, "kc_frac": kc_frac,
                         "seed": seed, "family": fam}
                    r.update(g)
                    rows.append(r)
                    all_rows.append(r)
            if rows:
                ma = float(np.mean([r["mean_active"] for r in rows if r["family"] == "REAL"]))
                empt = int(sum(r["empty_codes"] for r in rows if r["family"] == "REAL"))
                print_table(f"### maze {spec}: {n_states} states | kc_frac={kc_frac} "
                            f"(~{ma:.0f} active KCs/state, empty real codes over all seeds: {empt}) "
                            f"| {a.seeds} seeds | {time.time() - t0:.0f}s elapsed",
                            summarise(rows))

    if all_rows:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            wr.writeheader()
            wr.writerows(all_rows)
        print(f"\nWrote {out_path}")
    print(HOW_TO_READ)


if __name__ == "__main__":
    main()
