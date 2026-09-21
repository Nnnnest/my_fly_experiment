"""Does more RL self-play training actually make the connectome better at
tic-tac-toe, or does the decay found elsewhere in this project (grid-world
large maze, tic-tac-toe win-rate-vs-random/heuristic) mean more training
can make it WORSE? Win rate against one fixed opponent is noisy; this uses
full-coverage accuracy (does it pick a minimax-optimal move, checked
against literally every reachable state) as a cleaner, opponent-free
signal, checkpointed every CHECKPOINT_EVERY episodes through training.

Same self-play setup as play_tictactoe.py (periodic sleep included, since
that's what fixed self-play's decay). Writes
results/tictactoe_training_curve.csv and a matching plot, and prints a
recommendation for TRAIN_EPISODES in play_tictactoe.py based on where
accuracy actually peaks.
"""
import csv
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_THIS_DIR, os.path.dirname(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib.pyplot as plt

from envs.tictactoe_env import TicTacToe
from agents.connectome_tictactoe_agent import ConnectomeTicTacToeAgent
from run_tictactoe import sample_early_boards, EPSILON_START, EPSILON_END
from tictactoe_coverage_eval import full_coverage_accuracy
from tictactoe_solver import enumerate_reachable_states

TOTAL_EPISODES = 1500
CHECKPOINT_EVERY = 100
SLEEP_EVERY = 100
RESULTS_DIR = os.path.join(_THIS_DIR, "results")


def main():
    states = enumerate_reachable_states()  # computed once, reused at every checkpoint
    calibration_boards = sample_early_boards(300)

    a1 = ConnectomeTicTacToeAgent()
    a2 = ConnectomeTicTacToeAgent()
    a1.calibrate_baseline(calibration_boards)
    a2.calibrate_baseline(calibration_boards)

    epsilon_decay = (EPSILON_END / EPSILON_START) ** (1 / TOTAL_EPISODES)
    epsilon = EPSILON_START

    rows = []
    acc0, c0, t0n = full_coverage_accuracy(a1, states)
    print(f"episode 0 (untrained): full-coverage accuracy {acc0:.1f}% ({c0}/{t0n})")
    rows.append((0, acc0))

    t_start = time.time()
    for ep in range(1, TOTAL_EPISODES + 1):
        env = TicTacToe()
        env.reset()
        afterstates = {1: [], -1: []}
        a1_mark = 1 if ep % 2 == 1 else -1  # alternate who's X, same fairness fix as run_self_play
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
                result = {mark: (1 if outcome == 1 else 0), -mark: (-1 if outcome == 1 else 0)}
                a1.update_episode(afterstates[a1_mark], result[a1_mark])
                a2.update_episode(afterstates[-a1_mark], result[-a1_mark])
                break

        if SLEEP_EVERY and ep % SLEEP_EVERY == 0:
            a1.sleep()
            a2.sleep()
        epsilon *= epsilon_decay

        if ep % CHECKPOINT_EVERY == 0:
            acc, c, tot = full_coverage_accuracy(a1, states)
            elapsed = time.time() - t_start
            print(f"episode {ep}: full-coverage accuracy {acc:.1f}% ({c}/{tot})  [{elapsed:.0f}s elapsed]")
            rows.append((ep, acc))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "tictactoe_training_curve.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "full_coverage_accuracy_pct"])
        w.writerows(rows)
    print(f"\nwrote {csv_path}")

    episodes = [r[0] for r in rows]
    accs = [r[1] for r in rows]
    plt.figure()
    plt.plot(episodes, accs, marker="o")
    plt.xlabel("self-play training episodes")
    plt.ylabel("full-coverage accuracy (%)")
    plt.title("connectome: does more self-play training help?")
    plot_path = csv_path.replace(".csv", ".png")
    plt.savefig(plot_path)
    print(f"wrote {plot_path}")

    peak_i = accs.index(max(accs))
    peak_ep, peak_acc = episodes[peak_i], accs[peak_i]
    print(f"\ntrend: started at {accs[0]:.1f}%, ended at {accs[-1]:.1f}%, "
          f"peaked at {peak_acc:.1f}% at episode {peak_ep}")
    if accs[-1] < peak_acc - 2 and peak_ep < TOTAL_EPISODES:
        print(f"-> accuracy peaked BEFORE the end and declined afterward -- more "
              f"training made it worse here, consistent with the decay found "
              f"elsewhere in this project. Set TRAIN_EPISODES = {peak_ep} in "
              f"play_tictactoe.py instead of {TOTAL_EPISODES}, then delete "
              f"my_experiments/models/ to retrain with that value.")
    else:
        print("-> accuracy is still at/near its peak at the end of this run -- "
              "no evidence of decay up to this point. Try raising TOTAL_EPISODES "
              "here to see if it eventually declines too.")


if __name__ == "__main__":
    main()
