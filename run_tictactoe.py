"""Main experiment runner for the tic-tac-toe stage (spec.txt Stage 3,
Experiment 3). Runs each of the three agents (Q-learning, MLP, connectome)
against: a random opponent, a fixed heuristic opponent, and self-play.
Logs per-episode results to results/tictactoe_<agent>_<opponent>.csv for
view_tictactoe.py to summarize/plot.

All three agents share the SAME afterstate-value interface
(value/choose_action/update_episode), the SAME epsilon-decay schedule, and
the SAME forced-exploration floor (min_pulls) and Monte Carlo backup rule
-- carrying forward the fairness discipline from the bandit and grid-world
experiments (any assist given to one agent must be given to all three).

Run tictactoe_diagnostic.py first. This script assumes the encoding it
checks (board_odor_encoder.py's CELL_DRIVE/GROUP_SIZE) is already sane.
"""
import csv
import os
import random
import sys

# see tictactoe_diagnostic.py for why: covers fly_api.py living either
# inside my_experiments/ or one level up at the repo root
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_THIS_DIR, os.path.dirname(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from envs.tictactoe_env import TicTacToe
from opponents.opponents import random_opponent, heuristic_opponent
from agents.qlearning_tictactoe_agent import QLearningTicTacToeAgent
from agents.mlp_tictactoe_agent import MLPTicTacToeAgent
from agents.connectome_tictactoe_agent import ConnectomeTicTacToeAgent

N_EPISODES = 2000
EPSILON_START = 0.1
EPSILON_END = 0.01
EPSILON_DECAY = (EPSILON_END / EPSILON_START) ** (1 / N_EPISODES)  # multiplicative per episode, same as grid world
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def sample_early_boards(n=300, max_depth=4):
    """Random legal boards a few plies deep, for the connectome's
    calibrate_baseline() -- must be called on a clean, untrained network,
    so this is generated and used before any training happens anywhere in
    main()."""
    boards = set()
    env = TicTacToe()
    while len(boards) < n:
        env.reset()
        depth = random.randint(0, max_depth)
        for _ in range(depth):
            legal = env.legal_actions()
            if not legal:
                break
            env.step(random.choice(legal))
        boards.add(tuple(env.board))
    return list(boards)


def play_episode(agent, opponent_fn, agent_plays_first, epsilon):
    """Returns (afterstates, outcome) -- afterstates is the list of boards
    the agent itself produced (its own perspective each time), outcome is
    +1 win / -1 loss / 0 draw from the agent's perspective."""
    env = TicTacToe()
    env.reset()
    afterstates = []
    agent_mark = 1 if agent_plays_first else -1

    while True:
        if env.to_move == agent_mark:
            board_mine = env.board_from_perspective(agent_mark)
            legal = env.legal_actions()
            a = agent.choose_action(board_mine, legal, epsilon)
            _, outcome, done = env.step(a)
            afterstates.append(env.board_from_perspective(agent_mark))
            if done:
                return afterstates, (1 if outcome == 1 else 0)
        else:
            legal = env.legal_actions()
            board_theirs = env.board_from_perspective(env.to_move)
            a = opponent_fn(board_theirs, legal, env.to_move)
            _, outcome, done = env.step(a)
            if done:
                return afterstates, (-1 if outcome == 1 else 0)


def run_block(agent, agent_name, opponent_fn, opponent_name, n_episodes=N_EPISODES, sleep_every=None):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    suffix = "" if sleep_every is None else f"_sleep{sleep_every}"
    path = os.path.join(RESULTS_DIR, f"tictactoe_{agent_name}_{opponent_name}{suffix}.csv")
    epsilon = EPSILON_START
    has_drift = hasattr(agent, "drift_pct")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["episode", "agent_first", "outcome", "epsilon"]
        if has_drift:
            header.append("drift_pct")
        writer.writerow(header)
        for ep in range(n_episodes):
            agent_plays_first = (ep % 2 == 0)  # alternate so the agent learns both roles
            afterstates, outcome = play_episode(agent, opponent_fn, agent_plays_first, epsilon)
            agent.update_episode(afterstates, outcome)
            if sleep_every and (ep + 1) % sleep_every == 0 and hasattr(agent, "sleep"):
                agent.sleep()
            row = [ep, agent_plays_first, outcome, epsilon]
            if has_drift:
                row.append(agent.drift_pct())
            writer.writerow(row)
            epsilon *= EPSILON_DECAY
    print(f"wrote {path}")


def run_self_play(agent_cls, agent_name, calibration_boards=None, n_episodes=N_EPISODES, sleep_every=None):
    """Two fresh instances of the same agent class play each other; both
    learn from their own episode data. This is the spec's third opponent
    type ('another learning agent'). Who plays X (moves first) alternates
    by episode, same fairness discipline as run_block's agent_plays_first
    -- a fixed a1=X/a2=O assignment would give a1 a permanent structural
    first-move advantage and confound every result."""
    a1, a2 = agent_cls(), agent_cls()
    if calibration_boards is not None and hasattr(a1, "calibrate_baseline"):
        a1.calibrate_baseline(calibration_boards)
        a2.calibrate_baseline(calibration_boards)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    suffix = "" if sleep_every is None else f"_sleep{sleep_every}"
    path = os.path.join(RESULTS_DIR, f"tictactoe_{agent_name}_selfplay{suffix}.csv")
    epsilon = EPSILON_START
    has_drift = hasattr(a1, "drift_pct")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["episode", "a1_played_first", "outcome_a1", "epsilon"]
        if has_drift:
            header.append("drift_pct_a1")
        writer.writerow(header)
        for ep in range(n_episodes):
            env = TicTacToe()
            env.reset()
            afterstates = {1: [], -1: []}
            a1_played_first = (ep % 2 == 0)
            a1_mark = 1 if a1_played_first else -1
            agent_for_mark = {a1_mark: a1, -a1_mark: a2}

            while True:
                mark = env.to_move
                mover = agent_for_mark[mark]
                board_mine = env.board_from_perspective(mark)
                legal = env.legal_actions()
                a = mover.choose_action(board_mine, legal, epsilon)
                _, outcome, done = env.step(a)
                afterstates[mark].append(env.board_from_perspective(mark))
                if done:
                    # outcome is from the perspective of `mark`, who just moved
                    result = {mark: (1 if outcome == 1 else 0),
                              -mark: (-1 if outcome == 1 else 0)}
                    a1.update_episode(afterstates[a1_mark], result[a1_mark])
                    a2.update_episode(afterstates[-a1_mark], result[-a1_mark])
                    row = [ep, a1_played_first, result[a1_mark], epsilon]
                    if has_drift:
                        row.append(a1.drift_pct())
                    writer.writerow(row)
                    break
            if sleep_every and (ep + 1) % sleep_every == 0:
                if hasattr(a1, "sleep"):
                    a1.sleep()
                if hasattr(a2, "sleep"):
                    a2.sleep()
            epsilon *= EPSILON_DECAY
    print(f"wrote {path}")
    return a1, a2


def main():
    calibration_boards = sample_early_boards(300)

    agents = {
        "qlearning": QLearningTicTacToeAgent(),
        "mlp": MLPTicTacToeAgent(),
        "connectome": ConnectomeTicTacToeAgent(),
    }
    agents["connectome"].calibrate_baseline(calibration_boards)

    opponents = {"random": random_opponent, "heuristic": heuristic_opponent}
    for agent_name, agent in agents.items():
        for opponent_name, opponent_fn in opponents.items():
            run_block(agent, agent_name, opponent_fn, opponent_name)

    for agent_name, agent_cls in [
        ("qlearning", QLearningTicTacToeAgent),
        ("mlp", MLPTicTacToeAgent),
        ("connectome", ConnectomeTicTacToeAgent),
    ]:
        cb = calibration_boards if agent_name == "connectome" else None
        run_self_play(agent_cls, agent_name, calibration_boards=cb)


if __name__ == "__main__":
    main()
