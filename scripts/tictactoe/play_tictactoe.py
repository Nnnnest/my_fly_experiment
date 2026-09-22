"""Interactive terminal tic-tac-toe. Pick a tier, then an agent within it.

Tier "fair"     : plain agents, no symmetry, no cache -- matches the
                   original Tier-1 run_tictactoe.py output.
Tier "opt"      : SymmetryWrapper on all three agents (fair -- applied
                   equally) + MBValueCache for the connectome (speed only,
                   verified behavior-preserving via
                   scripts/tictactoe/verify_v2_matches_v1.py).
Tier "assisted" : opt's representation (symmetry+cache) PLUS a training
                   gate (error or visit) on the connectome only -- no
                   qlearning/mlp equivalent exists yet.
Tier "imitation": loads whichever models/tictactoe/connectome_imitation*
                   variants have been trained (train_imitation_tictactoe.py).

This trains its own SEPARATE, quick (TRAIN_EPISODES) demo weights per
tier/agent -- distinct from run_tictactoe.py's/run_tictactoe_tier2.py's
full 2000-episode benchmark runs. Delete models/tictactoe/*_demo_* to
force a retrain of the demo weights specifically (imitation weights are
untouched by that, since they don't share the _demo_ naming).
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))          # scripts/tictactoe
ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))     # my_experiments
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.tictactoe_env import TicTacToe
from agents.tictactoe.qlearning_tictactoe_agent import QLearningTicTacToeAgent
from agents.tictactoe.mlp_tictactoe_agent import MLPTicTacToeAgent
from agents.tictactoe.connectome_tictactoe_agent import ConnectomeTicTacToeAgent
from agents.tictactoe.connectome_tictactoe_agent_v2 import ConnectomeTicTacToeAgentV2
from agents.shared.symmetry_wrapper import SymmetryWrapper
from run_tictactoe import run_self_play, sample_early_boards

MODELS_DIR = os.path.join(ROOT, "models", "tictactoe")
TRAIN_EPISODES = 600
SLEEP_EVERY = 100

# tier -> {display_name: (factory_fn, needs_calibration_boards)}
# factory_fn takes no args and returns a fresh, untrained agent instance.
TIERS = {
    "fair": {
        "qlearning": (lambda: QLearningTicTacToeAgent(), False),
        "mlp": (lambda: MLPTicTacToeAgent(), False),
        "connectome": (lambda: ConnectomeTicTacToeAgent(), True),
    },
    "opt": {
        "qlearning_sym": (lambda: SymmetryWrapper(QLearningTicTacToeAgent()), False),
        "mlp_sym": (lambda: SymmetryWrapper(MLPTicTacToeAgent()), False),
        "connectome_sym_cache": (lambda: SymmetryWrapper(ConnectomeTicTacToeAgentV2(gate="none")), True),
    },
    "assisted": {
        "connectome_t2_error": (lambda: SymmetryWrapper(ConnectomeTicTacToeAgentV2(gate="error")), True),
        "connectome_t2_visit": (lambda: SymmetryWrapper(ConnectomeTicTacToeAgentV2(gate="visit")), True),
    },
}


def _save_paths(tier, name):
    prefix = os.path.join(MODELS_DIR, f"{name}_demo")
    if name.startswith("connectome"):
        return [prefix + "_weights.npz", prefix + "_baseline.npy"]
    if name.startswith("qlearning"):
        return [prefix + ".pkl"]
    return [prefix + ".pt"]


def _saved(tier, name):
    return all(os.path.exists(p) for p in _save_paths(tier, name))


def train_tier(tier):
    os.makedirs(MODELS_DIR, exist_ok=True)
    calibration_boards = sample_early_boards(300)
    agents = {}
    for name, (factory, needs_cal) in TIERS[tier].items():
        needs_sleep = "connectome" in name
        print(f"training {name} ({tier}) via self-play ({TRAIN_EPISODES} episodes"
              f"{', sleep every ' + str(SLEEP_EVERY) if needs_sleep else ''})...")
        a1, _ = run_self_play(
            factory, f"play_demo_{name}",
            calibration_boards=(calibration_boards if needs_cal else None),
            n_episodes=TRAIN_EPISODES,
            sleep_every=(SLEEP_EVERY if needs_sleep else None),
        )
        prefix = os.path.join(MODELS_DIR, f"{name}_demo")
        a1.save(prefix)
        agents[name] = a1
    return agents


def load_tier(tier):
    agents = {}
    for name, (factory, _) in TIERS[tier].items():
        agent = factory()
        agent.load(os.path.join(MODELS_DIR, f"{name}_demo"))
        agents[name] = agent
    return agents


def get_tier_agents(tier):
    if all(_saved(tier, name) for name in TIERS[tier]):
        print(f"loading saved '{tier}' models...")
        return load_tier(tier)
    return train_tier(tier)


def discover_imitation_agents():
    """Every models/tictactoe/connectome_imitation*_weights.npz found, with
    the two symmetry-trained variants correctly wrapped -- see
    eval_tier3_imitation.py's TAGS dict for why the wrapper matters here."""
    symmetry_tags = {"connectome_imitation_symmetry", "connectome_imitation_shuffle_symmetry"}
    found = {}
    if not os.path.isdir(MODELS_DIR):
        return found
    for fname in sorted(os.listdir(MODELS_DIR)):
        if not fname.endswith("_weights.npz") or not fname.startswith("connectome_imitation"):
            continue
        tag = fname[: -len("_weights.npz")]
        prefix = os.path.join(MODELS_DIR, tag)
        if not os.path.exists(prefix + "_baseline.npy"):
            continue
        base = ConnectomeTicTacToeAgent()
        base.load(prefix)
        found[f"imitation:{tag}"] = SymmetryWrapper(base) if tag in symmetry_tags else base
    return found


def render(board):
    sym = {1: "X", -1: "O", 0: None}
    cells = [sym[v] if v != 0 else str(i) for i, v in enumerate(board)]
    return "\n".join(" ".join(cells[r * 3:r * 3 + 3]) for r in range(3))


def play_one_game(agent, human_first):
    env = TicTacToe()
    env.reset()
    human_mark = 1 if human_first else -1
    print(f"\nyou are {'X' if human_mark == 1 else 'O'}. cells are numbered 0-8, "
          f"left-to-right, top-to-bottom.\n")

    while True:
        print(render(env.board))
        legal = env.legal_actions()
        if env.to_move == human_mark:
            a = None
            while a not in legal:
                raw = input(f"your move {legal}: ").strip()
                if raw.isdigit() and int(raw) in legal:
                    a = int(raw)
                else:
                    print("invalid move, try again")
        else:
            board_theirs = env.board_from_perspective(env.to_move)
            a = agent.best_action(board_theirs, legal)
            print(f"agent plays {a}")
        _, outcome, done = env.step(a)
        if done:
            print(render(env.board))
            if outcome == 0:
                print("draw!")
            else:
                winner_mark = -env.to_move
                print("you win!" if winner_mark == human_mark else "agent wins!")
            break


def main():
    print("tiers: fair (no assist), opt (symmetry+cache), assisted (symmetry+cache+gate)")
    tier = None
    while tier not in TIERS:
        tier = input("pick a tier: ").strip().lower()

    agents = get_tier_agents(tier)
    agents.update(discover_imitation_agents())
    if not any(k.startswith("imitation:") for k in agents):
        print("(no imitation-trained model found -- run train_imitation_tictactoe.py to add one)")

    while True:
        print("\nagents: " + ", ".join(agents.keys()))
        choice = input("play against which agent? (or 'tier' to switch tier, 'quit'): ").strip().lower()
        if choice == "quit":
            break
        if choice == "tier":
            main()
            return
        if choice not in agents:
            print("unknown agent name")
            continue
        first = input("go first? (y/n): ").strip().lower().startswith("y")
        play_one_game(agents[choice], human_first=first)


if __name__ == "__main__":
    main()
