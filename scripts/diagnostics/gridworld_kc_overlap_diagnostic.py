"""K-hot code overlap diagnostic for the 113-state grid-world maze
(SparseStateActionEncoder), closing the open item in notes/
summary_addendum.txt section 6: whether k-hot code overlap/interference,
not just the missing-decay mechanism, explains why the rank-gated agent's
policy accuracy plateaus at ~0.80 instead of reaching 1.0.

Reuses MBValueCache.overlap_stats() directly (same object the agent
itself builds at construction) plus a second, more targeted view:
WITHIN-STATE overlap -- the 4 actions available at the SAME state, which
are exactly the codes whose relative VALUES decide the greedy policy
there. High overlap among those specifically (vs. the overall/random
pairwise number) is what would explain persistent ranking conflicts,
as opposed to overlap between unrelated states, which mostly just costs
some cross-state training interference.
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_THIS_DIR, os.path.dirname(_THIS_DIR), os.path.dirname(os.path.dirname(_THIS_DIR))):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import numpy as np

from fly_api import FlyBrainAPI
from agents.shared.paths import FLY_ROOT
from agents.shared.sparse_encoder import SparseStateActionEncoder
from agents.shared.mb_value_cache import MBValueCache
from envs.maze_generator import generate_maze
from envs.gridworld_env import GridWorld

MAZE_W, MAZE_H = 19, 13   # same maze as run_gridworld_large_v2.py's default
N_ACTIONS = 4


def main():
    maze = generate_maze(MAZE_W, MAZE_H, extra_connections=0.1, seed=0)
    env = GridWorld(grid=maze, max_steps=350)
    trav_ids = [env.state_id((r, c)) for r in range(env.height)
                for c in range(env.width) if not env._is_wall(r, c)]
    n_states = len(trav_ids)
    print(f"maze: {MAZE_W}x{MAZE_H}, traversable states={n_states}, pairs={n_states*N_ACTIONS}")

    live_path = os.path.join(FLY_ROOT, "my_experiments", "results", "gridworld", "live_alpn_indices.npy")
    encoder = SparseStateActionEncoder(n_states, N_ACTIONS, live_path, k=6, max_overlap=1, seed=0)

    brain = FlyBrainAPI(mode="mb", path=FLY_ROOT)
    cache = MBValueCache(brain, encoder)

    print("\n1. Overall pairwise overlap (all (state,action) pairs -- the number")
    print("   already computed at agent construction time, reproduced standalone):")
    stats = cache.overlap_stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")

    print("\n2. Within-state overlap (the 4 actions at the SAME state -- what the")
    print("   greedy policy actually compares to pick an action):")
    B = (cache.codes > 0).astype(np.float32)
    within = []
    for s in range(n_states):
        ids = [s * N_ACTIONS + a for a in range(N_ACTIONS)]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                bi, bj = B[ids[i]], B[ids[j]]
                inter = float((bi * bj).sum())
                union = float(bi.sum() + bj.sum() - inter)
                within.append(inter / union if union > 0 else 0.0)
    within = np.array(within)
    print(f"   mean_jaccard={within.mean():.3f} p95={np.quantile(within, 0.95):.3f} "
          f"frac>0.2={float((within > 0.2).mean()):.3f} (n_pairs={len(within)})")

    print(f"\n   comparison: within-state mean {within.mean():.3f} vs "
          f"overall pairwise mean {stats['jaccard_mean']:.3f}")
    if within.mean() > stats['jaccard_mean'] * 1.2:
        print("   -> within-state pairs overlap MORE than random pairs: the actions")
        print("      that need to be told apart for a correct policy are exactly the")
        print("      ones the encoding separates LEAST well -- direct evidence for the")
        print("      k-hot-overlap-causes-persistent-ranking-conflicts hypothesis.")
    else:
        print("   -> within-state overlap is NOT elevated relative to random pairs --")
        print("      the plateau is more likely the missing-decay/gate-tuning mechanism")
        print("      than code-level interference specifically.")


if __name__ == "__main__":
    main()
