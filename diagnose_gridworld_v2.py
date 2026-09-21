"""Pre-flight diagnostic for the v2 approach (~1-3 min). Run BEFORE the large maze.

Run from the directory that contains mb_circuit.npz:
    python diagnose_v2.py [--states 113] [--k 6] [--max-overlap 1]

Checks, in order (each prints PASS/WARN/FAIL):
  1. cache is exact          - fast values == brain.step() on the real circuit
  2. KC order                - brain.KC sorted? (library indexing quirk)
  3. live ALPN scan          - analytic scan vs your saved live_alpn_indices.npy
  4. code overlap            - how much different (state,action) codes collide
  5. interference probe      - train one pair: own effect vs effect on all others
  6. retention probe         - does a learned value survive noisy training of others
  7. small-maze regression   - 26-state maze must still learn like the old agent
"""
import sys, os, time, argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "agents"))
sys.path.append(os.path.join(SCRIPT_DIR, ".."))  # fly_api lives one level up; append so it never shadows local modules
from fly_api import FlyBrainAPI
from sparse_encoder import SparseStateActionEncoder
from state_action_odor_encoder import StateActionOdorEncoder
from mb_value_cache import MBValueCache, train_no_readout, ensure_live_hops1, calibrate_own_effect

LIVE1 = os.path.join(SCRIPT_DIR, "live_alpn_indices.npy")


def tag(ok, warn=False):
    return "PASS" if ok else ("WARN" if warn else "FAIL")


def interference_probe(brain, enc, n_probe=15, seed=0):
    """Train ONE pair with reward; compare its own value change with the mean
    absolute change it causes on every other pair. selectivity = own / cross."""
    cache = MBValueCache(brain, enc)
    w0 = brain.wM.copy()
    base = cache.values_all().copy()
    rng = np.random.default_rng(seed)
    n = len(enc.patterns)
    own, cross = [], []
    for i in rng.choice(n, size=min(n_probe, n), replace=False):
        train_no_readout(brain, enc.patterns[i], hops=1, reward=1.0)
        cache.refresh()
        d = cache.values_all() - base
        mask = np.ones(n, bool); mask[i] = False
        own.append(abs(d[i])); cross.append(np.abs(d[mask]).mean())
        brain.wM[:] = w0
    return dict(own=float(np.mean(own)), cross=float(np.mean(cross)),
                selectivity=float(np.mean(own) / max(np.mean(cross), 1e-12)))


def retention_probe(brain, enc, n_noise=200, n_targets=5, seed=0):
    """Teach pair A (3 reward trains), then apply n_noise random reward/punish
    trains to OTHER pairs (mimics exploration noise). retention = fraction of
    A's learned value change that is still there."""
    cache = MBValueCache(brain, enc)
    w0 = brain.wM.copy()
    rng = np.random.default_rng(seed)
    n = len(enc.patterns)
    ret = []
    for A in rng.choice(n, size=min(n_targets, n), replace=False):
        v0 = cache.values_pairs([A])[0]
        for _ in range(3):
            train_no_readout(brain, enc.patterns[A], hops=1, reward=1.0)
        cache.refresh()
        v1 = cache.values_pairs([A])[0]
        others = [j for j in rng.choice(n, size=min(n_noise, n), replace=True) if j != A]
        for j in others:
            kw = dict(reward=1.0) if rng.random() < 0.5 else dict(punish=1.0)
            train_no_readout(brain, enc.patterns[j], hops=1, **kw)
        cache.refresh()
        v2 = cache.values_pairs([A])[0]
        gain = v1 - v0
        if abs(gain) > 1e-9:
            ret.append((v2 - v0) / gain)
        brain.wM[:] = w0
    return dict(retention_mean=float(np.mean(ret)) if ret else float("nan"),
                retention_min=float(np.min(ret)) if ret else float("nan"))


def build_task(brain, live, maze=(19, 13), k=6, max_overlap=1, seed=0):
    """Maze -> labelled (state, action) task + k-hot codes + value cache."""
    from envs.gridworld_env import GridWorld, ACTIONS
    from envs.maze_generator import generate_maze
    env = GridWorld(grid=generate_maze(maze[0], maze[1], extra_connections=0.1, seed=0))
    bfs = env.bfs_distances_from_goal()
    trav = [env.state_id(p) for p in env.traversable_states()]
    ns = len(trav)
    enc = SparseStateActionEncoder(ns, 4, live, k=k, max_overlap=max_overlap, seed=seed)
    cache = MBValueCache(brain, enc)
    good = np.zeros((ns, 4), bool)
    for si, sid in enumerate(trav):
        r, c = divmod(sid, env.width)
        for a, (dr, dc) in ACTIONS.items():
            nr, nc = r + dr, c + dc
            nsid = sid if env._is_wall(nr, nc) else env.state_id((nr, nc))
            good[si, a] = bfs[nsid] < bfs[sid]
    active = [si for si, sid in enumerate(trav) if divmod(sid, env.width) != env.goal]
    return dict(env=env, trav=trav, ns=ns, enc=enc, cache=cache, good=good, active=active)


def ceiling_test(task, epochs=300, seed=0):
    """REPRESENTATION CEILING. Fit an unconstrained linear readout on the fixed KC codes
    with a ranking perceptron (no plasticity-rule constraints, no weight bounds). If even
    this cannot rank the correct action first in every state, the KC representation itself
    cannot hold this task; if it can, the limit is the sign-gated learning rule."""
    codes, good, active = task["cache"].codes.astype(np.float64), task["good"], task["active"]
    w = np.zeros(codes.shape[1])
    rng = np.random.default_rng(seed)

    def wrong(si, w):
        v = codes[si * 4:(si + 1) * 4] @ w
        g, b = good[si], ~good[si]
        return (v[g].max() - v[b].max() <= 1e-9) if b.any() else False, v

    for ep in range(1, epochs + 1):
        errs = 0
        for si in rng.permutation(active):
            bad, v = wrong(si, w)
            if bad:
                g = np.where(good[si])[0]; b = np.where(~good[si])[0]
                a_good = g[np.argmax(v[g])]; a_bad = b[np.argmax(v[b])]
                w += codes[si * 4 + a_good] - codes[si * 4 + a_bad]
                errs += 1
        if errs == 0:
            break
    acc = float(np.mean([not wrong(si, w)[0] for si in active]))
    return dict(accuracy=acc, epochs=ep, converged=errs == 0)


def sweep_test(brain, live, n_sweeps=30, k=6, max_overlap=1, margin_frac=1.0, maze=(19, 13), seed=0,
               configs=None, task=None, quiet=False):
    """Maze-free plasticity test. Train every (state, action) pair with its CORRECT
    label (reward if the move gets closer to the goal, else punish), sweep after
    sweep, and track policy accuracy = fraction of states whose greedy action is a
    correct move. Compares always-train ('none') with the error-driven gate
    ('error': train only while the pair's value is on the wrong side of a margin).
    Isolates the learning rule from exploration/loop dynamics."""
    t = task or build_task(brain, live, maze=maze, k=k, max_overlap=max_overlap, seed=seed)
    ns, enc, cache, good, active = t["ns"], t["enc"], t["cache"], t["good"], t["active"]
    w0 = brain.wM.copy()
    cache.refresh()  # task/cache may be reused: values must reflect the CURRENT (initial) weights
    base = cache.values_all().reshape(ns, 4).copy()
    d_r, d_p = calibrate_own_effect(brain, cache, enc)
    tau_r, tau_p = margin_frac * d_r, margin_frac * d_p
    pairs = [(si, a) for si in active for a in range(4)]

    def acc():
        cache.refresh()
        vals = cache.values_all().reshape(ns, 4) - base
        return float(np.mean([good[si, int(np.argmax(vals[si]))] for si in active]))

    if not quiet:
        print(f"    maze states={ns}, one-train own effect: reward={d_r:.2f} punish={d_p:.2f} "
              f"(margins {tau_r:.2f}/{tau_p:.2f})")
    show = {1, 2, 3, 5, 10, 15, 20, 30, 40, 60}
    out = {}
    configs = configs or [("always", "none", 0.0), ("error", "error", 0.0),
                          ("rank m=0.02", "rank", 0.0, 0.02), ("rank m=0.05", "rank", 0.0, 0.05),
                          ("rank m=0.5", "rank", 0.0, 0.5)]
    for cfg in configs:
        label, gate, sleep_rate = cfg[:3]
        mf = cfg[3] if len(cfg) > 3 else margin_frac
        tau_r, tau_p = mf * d_r, mf * d_p
        brain.wM[:] = w0; brain.ref_mb[:] = 0; np.add.at(brain.ref_mb, brain.km_mi, brain.wM[brain.km_mask].astype(float))
        brain._km_tag = set(); cache.refresh()
        rng = np.random.default_rng(seed)
        hist, tot = [], 0
        for sw in range(1, n_sweeps + 1):
            n_tr = 0
            for j in rng.permutation(len(pairs)):
                si, a = pairs[j]
                pid = si * 4 + a
                if gate == "error":
                    cache.refresh()
                    v = cache.values_pairs([pid])[0] - base[si, a]
                    if (good[si, a] and v >= tau_r) or ((not good[si, a]) and v <= -tau_p):
                        continue
                elif gate == "rank":
                    # satisfied when the ranking at this state is already right by a margin:
                    # a rewarded action must beat all others; a punished one must lose to some other
                    cache.refresh()
                    vals = cache.values_state(si) - base[si]
                    m = vals[a] - np.max(np.delete(vals, a))
                    if (good[si, a] and m >= tau_r) or ((not good[si, a]) and m <= -tau_p):
                        continue
                kw = dict(reward=1.0) if good[si, a] else dict(punish=1.0)
                train_no_readout(brain, enc.patterns[pid], hops=1, **kw)
                n_tr += 1
            tot += n_tr
            if sleep_rate > 0:
                brain.sleep(episodes=1, rate=sleep_rate)
            hist.append((sw, acc(), n_tr))
        out[label] = hist
        if not quiet:
            print(f"    {label:11s} " + "  ".join(f"s{sw}:{a:.2f}({n})" for sw, a, n in hist if sw in show)
              + f"   total trains={tot}")
    brain.wM[:] = w0
    return out


def small_maze_regression(live, episodes=60, seed=0):
    from envs.gridworld_env import GridWorld
    from agents.connectome_gridworld_agent_v2 import ConnectomeGridAgentV2
    cfgs = {"onehot (no decay)": dict(encoding="onehot", train_decay=False),
            "khot (no decay)": dict(encoding="khot", train_decay=False),
            "khot + decay": dict(encoding="khot", train_decay=True),
            "khot + error gate": dict(encoding="khot", gate="error"),
            "khot + rank gate": dict(encoding="khot", gate="rank")}
    env = GridWorld(seed=seed)
    bfs = env.bfs_distances_from_goal()
    trav = [env.state_id(p) for p in env.traversable_states()]
    out = {}
    for name, cfg in cfgs.items():
        agent = ConnectomeGridAgentV2(trav, 4, LIVE1, bfs, hops=1, seed=seed, **cfg)
        succ, t0, steps = [], time.time(), 0
        for ep in range(episodes):
            s = env.reset(); done = False
            while not done:
                a = agent.select_action(s)
                s2, r, done = env.step(a)
                agent.update(s, a, r, s2, done); s = s2; steps += 1
            agent.decay_epsilon(); succ.append(env.pos == env.goal)
        out[name] = dict(last20=float(np.mean(succ[-20:])), ms_per_step=1000 * (time.time() - t0) / steps)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=int, default=113, help="traversable states of the target maze")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--max-overlap", type=int, default=1)
    ap.add_argument("--sweeps", type=int, default=30, help="supervised sweeps in check [8]; 0 = skip")
    args = ap.parse_args()
    n_pairs = args.states * 4

    brain = FlyBrainAPI(mode="mb")
    live = ensure_live_hops1(LIVE1, brain)
    print(f"live ALPN positions (hops=1): {len(live)}")
    kh = SparseStateActionEncoder(args.states, 4, live, k=args.k, max_overlap=args.max_overlap)
    cache = MBValueCache(brain, kh)

    print("\n[1] cache exactness vs brain.step()")
    v = cache.verify(n=40)
    print(f"    max_abs_err={v['max_abs_err']:.2e} scale={v['scale']:.3f}  -> {tag(v['ok'])}")

    print("[2] KC ordering")
    print(f"    brain.KC sorted: {v['kc_sorted']}  -> {tag(v['kc_sorted'], warn=True)}"
          + ("" if v["kc_sorted"] else "  (library pairs sorted-rank weights with unsorted activations; check)"))

    print("[3] live ALPN scan (analytic vs saved file)")
    scan = cache.scan_live()
    print(f"    analytic={len(scan)} file={len(live)} identical={np.array_equal(scan, live)} "
          f"-> {tag(np.array_equal(scan, live), warn=True)}")

    print(f"[4] KC-code overlap for {n_pairs} pairs (lower = less interference)")
    encs = {f"k-hot k={args.k}": kh}
    if n_pairs <= len(live):
        encs["one-hot"] = StateActionOdorEncoder(args.states, 4, LIVE1)
    else:
        print(f"    one-hot needs {n_pairs} live ALPNs but only {len(live)} exist -> cannot be used at hops=1")
    for nm, e in encs.items():
        st = MBValueCache(brain, e).overlap_stats()
        print(f"    {nm:12s} jaccard mean={st['jaccard_mean']:.3f} p95={st['jaccard_p95']:.3f} "
              f"pairs>0.2={st['frac_pairs_jaccard_gt_0.2']:.3f} empty={st['empty_codes']} "
              f"active_KC={st['mean_kc_active']:.0f}")

    print("[5] interference probe (selectivity = effect on trained pair / mean effect on others; higher = better)")
    for nm, e in encs.items():
        r = interference_probe(brain, e)
        print(f"    {nm:12s} own={r['own']:.4f} cross={r['cross']:.5f} selectivity={r['selectivity']:.1f}")

    print("[6] retention probe (fraction of a learned value left after 200 noisy trains of other pairs)")
    for nm, e in encs.items():
        r = retention_probe(brain, e)
        print(f"    {nm:12s} retention mean={r['retention_mean']:.2f} min={r['retention_min']:.2f}")

    print("[7] small-maze regression (26 states, 60 episodes; old agent reached ~99% by ep 10-20)")
    for nm, r in small_maze_regression(live).items():
        ok = r["last20"] >= 0.9
        print(f"    {nm:18s} success(last20)={r['last20']:.2f} {r['ms_per_step']:.1f} ms/step -> {tag(ok, warn=True)}")

    if args.sweeps > 0:
        print("[8] supervised sweep: policy accuracy vs sweeps (accuracy(trains in that sweep)); 'error'/'rank' = error-driven gates")
        task = build_task(brain, live, k=args.k, max_overlap=args.max_overlap)
        cl = ceiling_test(task)
        print(f"    representation ceiling (unconstrained linear readout on KC codes): accuracy={cl['accuracy']:.2f} "
              f"({'converged' if cl['converged'] else 'not converged'} in {cl['epochs']} epochs)")
        sweep_test(brain, live, n_sweeps=args.sweeps, k=args.k, max_overlap=args.max_overlap, task=task)

    print("\nRead-out for [8]: if 'always' decays but 'rank'/'error' hold, use the error-driven gates. "
          "If ALL online schemes decay, compare with the representation ceiling: ceiling high = the KC codes can hold the "
          "task but the sign-gated rule cannot find the solution (a different circuit would not fix this; a different "
          "readout/learning rule might); ceiling low = the KC representation itself is at capacity. "
          "Run capacity_probe_v2.py to see where each breaks as the maze grows.\n"
          "Other checks: if [1] fails, do not trust any v2 result. If [7] fails for 'onehot (no decay)', the v2 machinery "
          "broke something that used to work. If [5]/[6] show k-hot is not more selective / retentive than one-hot, the "
          "encoding change will not help.")


if __name__ == "__main__":
    main()
