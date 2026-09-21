"""Interactive terminal tic-tac-toe: play against any of the three
RL-trained agents, plus a fourth "imitation" option if you've run
train_imitation_tictactoe.py (trained on minimax-optimal moves for every
reachable state instead of via trial-and-error reward -- see that
script's docstring). Run this -- if no saved RL models exist in
my_experiments/models/, it trains all three via self-play (TRAIN_EPISODES
each, connectome with periodic sleep since that measurably helped
self-play -- see project notes) and saves them there; subsequent runs
load instantly instead of retraining. Delete my_experiments/models/ to
force a retrain (e.g. after changing TRAIN_EPISODES).

TRAIN_EPISODES was cut from an initial 1500 -- run
tictactoe_training_curve.py first if you want a data-driven number for
this instead of a guess; it checkpoints full-coverage accuracy through
training and tells you exactly where (if anywhere) it peaks before
declining.

This trains a SEPARATE, quicker set of weights than run_tictactoe.py's
full 2000-episode/3-opponent benchmark -- it's for play, not for the
Stage 3/4 comparison numbers.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ROOT = os.path.join(SCRIPT_DIR, "..", "..")
from envs.tictactoe_env import TicTacToe
from agents.gridworld.qlearning_tictactoe_agent import QLearningTicTacToeAgent
from agents.gridworld.mlp_tictactoe_agent import MLPTicTacToeAgent
from agents.gridworld.connectome_tictactoe_agent import ConnectomeTicTacToeAgent
from run_tictactoe import run_self_play, sample_early_boards

MODELS_DIR = os.path.join(ROOT, "models")
TRAIN_EPISODES = 600  # see tictactoe_training_curve.py for a data-driven value instead of this guess
SLEEP_EVERY = 100

MODEL_FILES = {
    "qlearning": [os.path.join(MODELS_DIR, "qlearning.pkl")],
    "mlp": [os.path.join(MODELS_DIR, "mlp.pt")],
    "connectome": [os.path.join(MODELS_DIR, "connectome_weights.npz"),
                   os.path.join(MODELS_DIR, "connectome_baseline.npy")],
}
IMITATION_FILES = [os.path.join(MODELS_DIR, "connectome_imitation_weights.npz"),
                    os.path.join(MODELS_DIR, "connectome_imitation_baseline.npy")]


def all_saved():
    return all(os.path.exists(p) for paths in MODEL_FILES.values() for p in paths)


def train_and_save():
    os.makedirs(MODELS_DIR, exist_ok=True)
    calibration_boards = sample_early_boards(300)

    print(f"training qlearning via self-play ({TRAIN_EPISODES} episodes)...")
    q1, _ = run_self_play(QLearningTicTacToeAgent, "play_demo_qlearning", n_episodes=TRAIN_EPISODES)
    q1.save(MODEL_FILES["qlearning"][0])

    print(f"training mlp via self-play ({TRAIN_EPISODES} episodes)...")
    m1, _ = run_self_play(MLPTicTacToeAgent, "play_demo_mlp", n_episodes=TRAIN_EPISODES)
    m1.save(MODEL_FILES["mlp"][0])

    print(f"training connectome via self-play ({TRAIN_EPISODES} episodes, "
          f"sleep every {SLEEP_EVERY})...")
    c1, _ = run_self_play(ConnectomeTicTacToeAgent, "play_demo_connectome",
                           calibration_boards=calibration_boards, n_episodes=TRAIN_EPISODES,
                           sleep_every=SLEEP_EVERY)
    c1.save(os.path.join(MODELS_DIR, "connectome"))

    return {"qlearning": q1, "mlp": m1, "connectome": c1}


def load_agents():
    q = QLearningTicTacToeAgent()
    q.load(MODEL_FILES["qlearning"][0])
    m = MLPTicTacToeAgent()
    m.load(MODEL_FILES["mlp"][0])
    c = ConnectomeTicTacToeAgent()
    c.load(os.path.join(MODELS_DIR, "connectome"))
    return {"qlearning": q, "mlp": m, "connectome": c}


def render(board):
    """Numbered when empty (so you know which index to type), marked
    X/O otherwise."""
    sym = {1: "X", -1: "O", 0: None}
    cells = []
    for i, v in enumerate(board):
        cells.append(sym[v] if v != 0 else str(i))
    rows = [" ".join(cells[r * 3:r * 3 + 3]) for r in range(3)]
    return "\n".join(rows)


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
                winner_mark = -env.to_move  # to_move already flipped past the mover inside step()
                print("you win!" if winner_mark == human_mark else "agent wins!")
            break


def load_imitation_agent_if_present():
    """Optional 4th agent -- only added if train_imitation_tictactoe.py
    has been run. Not auto-trained here (that's a separate, much slower
    script by design -- see its docstring)."""
    if all(os.path.exists(p) for p in IMITATION_FILES):
        agent = ConnectomeTicTacToeAgent()
        agent.load(os.path.join(MODELS_DIR, "connectome_imitation"))
        return agent
    return None


def main():
    if all_saved():
        print("loading saved models...")
        agents = load_agents()
    else:
        agents = train_and_save()

    imitation_agent = load_imitation_agent_if_present()
    if imitation_agent is not None:
        agents["imitation"] = imitation_agent
    else:
        print("(no imitation-trained model found -- run train_imitation_tictactoe.py "
              "to add it as an option here)")

    while True:
        print("\nagents: " + ", ".join(agents.keys()))
        choice = input("play against which agent? (or 'quit'): ").strip().lower()
        if choice == "quit":
            break
        if choice not in agents:
            print("unknown agent name")
            continue
        first = input("go first? (y/n): ").strip().lower().startswith("y")
        play_one_game(agents[choice], human_first=first)


if __name__ == "__main__":
    main()
