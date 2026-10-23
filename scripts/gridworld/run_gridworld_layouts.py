"""Gridworld on independently generated maze layouts (revision plan, priority 2).
Layout seed (maze topology) and agent seed (encoding, exploration, start sequence) are separate.
Task name = '<size>@<layout seed>_rs'. Layout 0 is the maze used in the main experiments (check: --layouts 0).
    python -m scripts.gridworld.run_gridworld_layouts --list-layouts          # traversable states per layout
    python -m scripts.gridworld.run_gridworld_layouts --smoke
    python -m scripts.gridworld.run_gridworld_layouts --workers 14 [--ungated] [--max-overlap 2]"""
from functools import lru_cache

from core import paths
from core.runner import RunSpec, common_parser, run_all
from envs.maze_generator import generate_maze, count_traversable
import scripts.gridworld.run_gridworld as rg

EXPERIMENT = "gridworld_layouts"
LAYOUTS = list(range(100, 110))
AGENT_SEEDS = [0, 1, 2]
SIZES = ["maze72", "maze113"]


@lru_cache(maxsize=None)
def layout_maze(name):
    base, layout = name.split("@")
    w, h = rg.MAZES[base][:2]
    return generate_maze(w, h, extra_connections=0.1, seed=int(layout))


rg.get_maze = layout_maze      # run_gridworld.run_one looks the maze up by name; '<size>@<layout>' now resolves here


def run_one(spec, log):        # wrapper so that spawned workers import THIS module (and therefore the patch above)
    return rg.run_one(spec, log)


def variants(ungated, max_overlap):
    kw = {"encoding": "khot", "max_overlap": max_overlap}
    v = [("qlearning", "fair", {"lr": rg.ALPHA_Q}), ("mlp", "fair", {"lr": rg.LR_MLP}),
         rg.conn("gate_error", gate="error", margin_frac=rg.MARGIN_ERROR, **kw),
         rg.conn("gate_rank", gate="rank", margin_frac=rg.MARGIN_RANK, **kw),
         ("kc_delta", "diagnostic", {"lr": rg.ALPHA_KC}), ("rand_delta", "diagnostic", {"lr": rg.ALPHA_RAND})]
    if ungated:                # slow: about 2,000 s (72 states) and 4,600 s (113 states) per run in the main experiments
        v.append(rg.conn("opt", gate="none", **kw))
    return v


def main():
    ap = common_parser(__doc__)
    ap.add_argument("--layouts", type=int, nargs="+", default=None, help="layout seeds (default 100..109)")
    ap.add_argument("--ungated", action="store_true", help="also run the ungated k-hot connectome (slow)")
    ap.add_argument("--max-overlap", type=int, default=1, help="k-hot pair overlap cap (main experiments: 1)")
    ap.add_argument("--list-layouts", action="store_true")
    args = ap.parse_args()
    if args.tuning:
        raise SystemExit("no tuning mode: this experiment reuses the constants chosen for the main gridworld runs")
    layouts = args.layouts or (LAYOUTS[:1] if args.smoke else LAYOUTS)
    if args.list_layouts:
        for size in SIZES:
            for layout in layouts:
                n = count_traversable(layout_maze(f"{size}@{layout}"))
                print(f"{size}@{layout}: {n} traversable states, {4 * n} (state, action) pairs")
        return
    sizes = ["maze72"] if args.smoke else SIZES
    seeds = args.seeds or ([0] if args.smoke else AGENT_SEEDS)
    specs = []
    for size in sizes:
        _, _, max_steps, episodes = rg.MAZES[size]
        episodes = 30 if args.smoke else episodes
        for layout in layouts:
            name = f"{size}@{layout}"
            for agent, tier, extra in variants(args.ungated, args.max_overlap):
                for seed in seeds:
                    cfg = {"maze": name, "start": "rs", "max_steps": max_steps, "episodes": episodes,
                           "gamma": rg.GAMMA, "min_pulls": rg.MIN_PULLS, **extra}
                    specs.append(RunSpec(EXPERIMENT, f"{name}_rs", agent, tier, seed, cfg))
    run_all(specs, run_one, args,
            cost=lambda s: s.config["episodes"] * s.config["max_steps"] * (10 if s.agent == "mlp" else 1))


if __name__ == "__main__":
    main()
