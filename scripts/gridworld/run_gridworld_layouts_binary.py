"""Independent maze layouts (the layout experiment of scripts/gridworld/run_gridworld_layouts.py) with Q-learning and
the MLP trained on the connectome's BINARY feedback (+1 if the move reaches the goal or reduces the shortest-path
distance, otherwise -1), exactly as in scripts/gridworld/run_gridworld_binary.py (whose constants ALPHA_Q_BIN and
LR_MLP_BIN are reused). Same layouts (100..109), sizes, agent seeds, episodes and start sequences as the layout
experiment, so results pair with runs/gridworld_layouts on (task, seed).

    python -m scripts.gridworld.run_gridworld_layouts_binary --list-layouts
    python -m scripts.gridworld.run_gridworld_layouts_binary --smoke
    python -m scripts.gridworld.run_gridworld_layouts_binary --workers 14
"""
from core import paths  # noqa: F401
from core.runner import RunSpec, common_parser, run_all
from envs.maze_generator import count_traversable
import scripts.gridworld.run_gridworld as rg
import scripts.gridworld.run_gridworld_binary as rb
import scripts.gridworld.run_gridworld_layouts as rl

EXPERIMENT = "gridworld_layouts_binary"
rb.get_maze = rl.layout_maze      # the binary runner looks the maze up by name; '<size>@<layout>' now resolves here


def run_one(spec, log):           # wrapper so that spawned workers import THIS module (and therefore the patch above)
    return rb.run_one(spec, log)


def main():
    ap = common_parser(__doc__)
    ap.add_argument("--layouts", type=int, nargs="+", default=None, help="layout seeds (default 100..109)")
    ap.add_argument("--list-layouts", action="store_true")
    args = ap.parse_args()
    if args.tuning:
        raise SystemExit("no tuning mode: constants come from the binary-feedback main experiment")
    layouts = args.layouts or (rl.LAYOUTS[:1] if args.smoke else rl.LAYOUTS)
    if args.list_layouts:
        for size in rl.SIZES:
            for layout in layouts:
                n = count_traversable(rl.layout_maze(f"{size}@{layout}"))
                print(f"{size}@{layout}: {n} traversable states")
        return
    sizes = ["maze72"] if args.smoke else rl.SIZES
    seeds = args.seeds or ([0] if args.smoke else rl.AGENT_SEEDS)
    specs = []
    for size in sizes:
        _, _, max_steps, episodes = rg.MAZES[size]
        episodes = 30 if args.smoke else episodes
        for layout in layouts:
            name = f"{size}@{layout}"
            for agent, tier, extra in rb.final_variants():
                for seed in seeds:
                    cfg = {"maze": name, "start": "rs", "max_steps": max_steps, "episodes": episodes,
                           "gamma": rg.GAMMA, "min_pulls": rg.MIN_PULLS, "feedback": "binary", **extra}
                    specs.append(RunSpec(EXPERIMENT, f"{name}_rs", agent, tier, seed, cfg))
    run_all(specs, run_one, args,
            cost=lambda s: s.config["episodes"] * s.config["max_steps"] * (10 if s.agent == "mlp" else 1))


if __name__ == "__main__":
    main()
