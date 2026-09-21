"""Large-maze experiment, v2.

Differences from run_gridworld_large.py (run_gridworld_lrage.py):
  * connectome agent is ConnectomeGridAgentV2 (k-hot codes, cached values,
    decaying train probability) and runs at hops=1 by default, so it is directly
    comparable with the 26/72-state results;
  * several connectome VARIANTS can run side by side (ablation), so a result can
    be attributed to the encoding, the training gate, or hops;
  * per-run collapse metrics (peak vs final rolling success) are printed and
    saved, because the failure you saw was "learns, then loses it";
  * baselines (Q-learning, MLP) are constructed and trained exactly as before.

Run from the directory that contains mb_circuit.npz (same as before).
  python run_gridworld_large_v2.py                       # default: khot_rank khot + baselines
  python run_gridworld_large_v2.py --agents khot_rank khot --seeds 0 --episodes 100   # quick look
  python run_gridworld_large_v2.py --agents onehot_h2 onehot_h2_decay             # slow, hops=2
"""
import sys, os, csv, time, argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "agents"))
from envs.gridworld_env import GridWorld, ACTIONS
from envs.maze_generator import generate_maze
from agents.qlearning_gridworld_agent import QLearningGridAgent
from agents.mlp_gridworld_agent import MLPGridAgent
from agents.connectome_gridworld_agent_v2 import ConnectomeGridAgentV2
from agents.connectome_kc_delta_agent import ConnectomeKCDeltaAgent
from mb_value_cache import ensure_live_hops1

GAMMA = 0.95
MIN_PULLS = 5
WINDOW = 20
LIVE = {1: os.path.join(SCRIPT_DIR, "live_alpn_indices.npy"),
        2: os.path.join(SCRIPT_DIR, "live_alpn_indices_hops2.npy")}

# name -> connectome configuration
# gate = WHEN train() is called: none (always) / visit (1/n by visit count) /
# error (own value vs margin) / rank (train only while the ranking at that state is wrong)
CONNECTOME_VARIANTS = {
    "khot_rank":       dict(encoding="khot",   hops=1, gate="rank"),   # error-driven policy learning
    "khot_rank02":     dict(encoding="khot",   hops=1, gate="rank", margin_frac=0.02),
    "khot_rank_cap":   dict(encoding="khot",   hops=1, gate="rank", max_pair_trains=10),
    "khot_error":      dict(encoding="khot",   hops=1, gate="error"),
    "khot_decay":      dict(encoding="khot",   hops=1, gate="visit"),
    "khot":            dict(encoding="khot",   hops=1, gate="none"),   # encoding change only
    # kind="delta": connectome KC code as a FIXED state representation + delta-rule Q readout
    # (readout weights are NOT connectome synapses; gets the same shaped reward as the baselines).
    # rand_delta is the control: same size/sparsity, random codes with no connectome structure.
    "kc_delta":        dict(kind="delta", features="kc"),
    "rand_delta":      dict(kind="delta", features="random"),
    "onehot_h2":       dict(encoding="onehot", hops=2, gate="none"),   # reproduces the old run
    "onehot_h2_decay": dict(encoding="onehot", hops=2, gate="visit"),
}
ALL_AGENTS = ["qlearning", "mlp"] + list(CONNECTOME_VARIANTS)


def shape_reward(raw_r, bfs_dist, s, s2, gamma=GAMMA):
    return raw_r + (bfs_dist[s] - gamma * bfs_dist[s2])


def live_path(hops):
    if hops == 1:
        ensure_live_hops1(LIVE[1])
    elif not os.path.exists(LIVE[2]):
        raise SystemExit(f"{LIVE[2]} missing (hops=2 live scan is not analytic; reuse your saved file)")
    return LIVE[hops]


def make_agent(name, traversable_ids, n_states, n_actions, bfs_dist, seed, args):
    if name == "qlearning":
        return QLearningGridAgent(n_states, n_actions, min_pulls=MIN_PULLS, seed=seed)
    if name == "mlp":
        return MLPGridAgent(n_states, n_actions, min_pulls=MIN_PULLS, seed=seed)
    cfg = CONNECTOME_VARIANTS[name]
    if cfg.get("kind") == "delta":
        return ConnectomeKCDeltaAgent(traversable_ids, n_actions, live_path(1), min_pulls=MIN_PULLS,
                                      seed=seed, k=args.k, max_overlap=args.max_overlap,
                                      alpha=args.alpha, features=cfg["features"], kc_frac=args.kc_frac)
    return ConnectomeGridAgentV2(
        traversable_ids, n_actions, live_path(cfg["hops"]), bfs_dist,
        hops=cfg["hops"], min_pulls=MIN_PULLS, seed=seed, encoding=cfg["encoding"],
        k=args.k, max_overlap=args.max_overlap, gate=cfg["gate"], margin_frac=cfg.get("margin_frac", args.margin_frac),
        max_pair_trains=cfg.get("max_pair_trains"),
        train_power=args.train_power, train_floor=args.train_floor)


def policy_accuracy(agent, env, trav_ids, bfs_dist):
    """Fraction of (non-goal) states whose greedy action moves closer to the goal.
    Direct read-out of the learned policy, independent of exploration noise."""
    ok = n = 0
    for sid in trav_ids:
        r, c = divmod(sid, env.width)
        if (r, c) == env.goal:
            continue
        dr, dc = ACTIONS[agent.greedy_action(sid)]
        nr, nc = r + dr, c + dc
        nsid = sid if env._is_wall(nr, nc) else env.state_id((nr, nc))
        ok += bfs_dist[nsid] < bfs_dist[sid]
        n += 1
    return ok / max(n, 1)


def summarize(successes, window=WINDOW):
    """Rolling-success metrics that expose 'learns, then collapses'."""
    x = np.asarray(successes, dtype=float)
    if len(x) < window:
        return dict(peak=float(x.mean()), final=float(x.mean()), first90=None, collapsed=False)
    roll = np.convolve(x, np.ones(window) / window, mode="valid")
    hit = np.where(roll >= 0.9)[0]
    peak, final = float(roll.max()), float(roll[-1])
    return dict(peak=peak, final=final,
                first90=int(hit[0]) + window if len(hit) else None,
                collapsed=bool(peak >= 0.5 and peak - final > 0.3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--agents", nargs="+", default=["qlearning", "mlp", "khot_rank", "khot"],
                    choices=ALL_AGENTS)
    ap.add_argument("--maze-w", type=int, default=19)
    ap.add_argument("--maze-h", type=int, default=13)
    ap.add_argument("--max-steps", type=int, default=350)
    ap.add_argument("--k", type=int, default=6, help="ALPNs per (state,action) code")
    ap.add_argument("--max-overlap", type=int, default=1, help="max shared ALPNs between two codes")
    ap.add_argument("--margin-frac", type=float, default=0.05,
                    help="error/rank gate margin, in units of one train's effect. Real-circuit sweep at 113 states: "
                         "0.02-0.05 held ~0.9+ policy accuracy, 0.5 and above decayed")
    ap.add_argument("--alpha", type=float, default=0.5, help="delta-rule step size for kc_delta / rand_delta")
    ap.add_argument("--kc-frac", type=float, default=0.05,
                    help="code sparsity for kc_delta / rand_delta (fraction of KCs active per state)")
    ap.add_argument("--train-power", type=float, default=1.0)
    ap.add_argument("--train-floor", type=float, default=0.05)
    ap.add_argument("--random-start", action="store_true",
                    help="start every episode from a random non-goal cell (same start sequence for every agent in a seed); "
                         "forces the WHOLE map to be learned, not just one path")
    ap.add_argument("--progress", type=int, default=20)
    ap.add_argument("--tag", default="large_v2")
    args = ap.parse_args()

    maze = generate_maze(args.maze_w, args.maze_h, extra_connections=0.1, seed=0)
    probe = GridWorld(grid=maze, max_steps=args.max_steps)
    n_states, n_actions = probe.n_states, 4
    n_trav = sum(1 for row in probe.grid for c in row if c != "#")
    print(f"=== LARGE MAZE v2: traversable={n_trav}, pairs={n_trav * 4}, max_steps={args.max_steps} ===")
    print(f"agents={args.agents} seeds={args.seeds} episodes={args.episodes}")

    results_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"gridworld_results_{args.tag}.csv")
    meta_path = os.path.join(results_dir, f"gridworld_results_{args.tag}_meta.csv")
    pol_path = os.path.join(results_dir, f"gridworld_results_{args.tag}_policy.csv")

    with open(csv_path, "w", newline="") as f, open(meta_path, "w", newline="") as fm, \
            open(pol_path, "w", newline="") as fp:
        w, wm, wp = csv.writer(f), csv.writer(fm), csv.writer(fp)
        wp.writerow(["maze", "agent", "seed", "episode", "policy_acc", "cum_trained", "cum_skipped"])
        # same 7 columns as the old CSV plus optimal_steps (shortest path from that episode's start)
        w.writerow(["maze", "agent", "seed", "episode", "steps", "total_reward", "reached_goal", "optimal_steps"])
        wm.writerow(["maze", "agent", "seed", "seconds", "peak20", "final20", "first_ep_ge90",
                     "collapsed", "n_trained", "n_skipped", "cache_used", "top10_share", "max_pair_trains"])

        for seed in args.seeds:
            env = GridWorld(grid=maze, max_steps=args.max_steps, seed=seed)
            bfs_dist = env.bfs_distances_from_goal()
            trav_ids = [env.state_id((r, c)) for r in range(env.height)
                        for c in range(env.width) if not env._is_wall(r, c)]
            start_cells = [p for p in env.traversable_states() if p != env.goal]
            if seed == args.seeds[0]:
                print(f"  shortest path from the fixed start to the goal = {int(bfs_dist[env.state_id(env.start)])} steps"
                      f"{'; episodes use RANDOM starts' if args.random_start else ''}")

            for name in args.agents:
                try:
                    agent = make_agent(name, trav_ids, n_states, n_actions, bfs_dist, seed, args)
                except (AssertionError, RuntimeError) as e:
                    print(f"  [{name} seed={seed}] SKIPPED: {e}")
                    continue
                is_conn = name in CONNECTOME_VARIANTS
                is_gate = is_conn and CONNECTOME_VARIANTS[name].get("kind") != "delta"
                has_policy = is_conn and (getattr(agent, "cache", None) is not None or not is_gate)
                if is_gate and getattr(agent, "cache", None) is not None:
                    st = agent.cache.overlap_stats()
                    print(f"  [{name} seed={seed}] code overlap: jaccard mean={st['jaccard_mean']:.3f} "
                          f"p95={st['jaccard_p95']:.3f} empty={st['empty_codes']}")

                t0, succ = time.time(), []
                start_rng = np.random.default_rng(1000 + seed)  # identical start sequence for every agent
                for ep in range(args.episodes):
                    s, total_r = env.reset(), 0.0
                    if args.random_start:
                        pos = start_cells[int(start_rng.integers(len(start_cells)))]
                        env.set_pos(pos)
                        s = env.state_id(pos)
                    opt = int(bfs_dist[s])
                    for t in range(env.max_steps):
                        a = agent.select_action(s)
                        s2, r, done = env.step(a)
                        if is_gate:
                            agent.update(s, a, r, s2, done)           # raw reward, as before
                        else:
                            agent.update(s, a, shape_reward(r, bfs_dist, s, s2), s2, done)
                        s, total_r = s2, total_r + r
                        if done:
                            break
                    agent.decay_epsilon()
                    reached = env.pos == env.goal
                    succ.append(reached)
                    w.writerow([args.tag, name, seed, ep, t + 1, total_r, reached, opt])
                    f.flush()
                    if has_policy:
                        wp.writerow([args.tag, name, seed, ep,
                                     f"{policy_accuracy(agent, env, trav_ids, bfs_dist):.4f}",
                                     agent.n_trained, getattr(agent, "n_skipped", 0)])
                        fp.flush()
                    if (ep + 1) % args.progress == 0 or ep == args.episodes - 1:
                        el = time.time() - t0
                        rate = np.mean(succ[-args.progress:])
                        eta = el / (ep + 1) * (args.episodes - ep - 1)
                        extra = ""
                        if has_policy:
                            extra = (f" policy_acc={policy_accuracy(agent, env, trav_ids, bfs_dist):.2f}"
                                     f" trained={agent.n_trained}")
                        print(f"    [{name} seed={seed}] ep {ep+1}/{args.episodes} "
                              f"recent_success={rate:.2f}{extra} elapsed={el:.0f}s eta={eta:.0f}s")

                secs = time.time() - t0
                sm = summarize(succ)
                n_tr = getattr(agent, "n_trained", "")
                n_sk = getattr(agent, "n_skipped", "")
                used = getattr(agent, "cache", None) is not None if is_gate else ""
                tc = np.sort(agent.train_counts.ravel())[::-1] if is_gate else None
                top10 = f"{tc[:10].sum() / max(tc.sum(), 1):.3f}" if is_gate else ""
                maxpair = int(tc[0]) if is_gate else ""
                wm.writerow([args.tag, name, seed, f"{secs:.1f}", f"{sm['peak']:.3f}",
                             f"{sm['final']:.3f}", sm["first90"], sm["collapsed"], n_tr, n_sk, used, top10, maxpair])
                fm.flush()
                print(f"  {name} seed={seed} done in {secs:.0f}s | peak20={sm['peak']:.2f} "
                      f"final20={sm['final']:.2f} first>=90%={sm['first90']} "
                      f"{'COLLAPSED' if sm['collapsed'] else 'stable'}"
                      + (f" | trained={n_tr} skipped={n_sk} cache={used} top10pairs={top10} max_pair={maxpair}" if is_gate else ""))
    print(f"Wrote {csv_path}\nWrote {meta_path}\nWrote {pol_path}")


if __name__ == "__main__":
    main()
