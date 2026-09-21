"""Imitation-learning comparison: instead of learning tic-tac-toe through
trial-and-error reward signals (the RL approach used by run_tictactoe.py
and everywhere else in this project so far), directly show the connectome
the minimax-optimal move for EVERY reachable state.

FIRST RUN RESULT (all three toggles below OFF): full-coverage accuracy
53.8%, barely different from the UNTRAINED baseline of 54.7% -- i.e. one
full pass of dense training, with no rest, washed out almost everything
it should have taught. That's stronger and more specific evidence for a
retention/interference problem than anything found so far (RL self-play
WITH periodic sleep reached a real 66-68%, with much less/noisier signal
per state -- consolidation appears to matter more than signal quality).

Three independent, toggleable fixes to isolate which one(s) actually
matter -- change ONE at a time and compare against the 53.8% baseline
above before turning more on at once, or you won't know which fix did it:

  SHUFFLE_STATES: enumerate_reachable_states() visits states in a fixed
    depth-first order (all continuations of one opening move cluster
    together before the next). States trained early have ~4,000
    subsequent training calls to be overwritten by -- the same kind of
    recency confound already found and fixed for MLP's replay buffer in
    tictactoe_defense_diagnostic.py. Shuffling the state VISIT order (not
    just examples within a state, which was already random) removes it.

  SLEEP_EVERY_N_STATES: the imitation loop currently calls sleep() zero
    times across ~27,000 train() calls. RL self-play's biggest single fix
    was periodic sleep -- untested here until now.

  USE_SYMMETRY: wraps the agent in SymmetryWrapper (tictactoe_symmetry.py)
    so all 8 rotations/mirrors of a board collapse onto one canonical
    encoding -- ~4,520 raw states become ~765 canonical ones, directly
    reducing how many distinct patterns compete for capacity/weights.

This is a genuinely different training paradigm from the rest of the
project (imitation, not reinforcement) -- label it as such in any
write-up, not as more RL-comparison evidence.

Prints a timing estimate after the first ESTIMATE_AFTER states, so you can
judge whether to let it finish or Ctrl-C and fall back to the grid-world
maze instead (far fewer states -- 26/72 -- if tic-tac-toe's ~4,500
decision states turn out to be too slow given train()'s per-call cost).

After training, evaluates FULL-COVERAGE accuracy (every state it was
trained on -- this measures memorization, not generalization, by design)
using the same tictactoe_coverage_eval.full_coverage_accuracy() that
evaluate_rl_coverage.py uses on the RL-trained connectome. Saves the
result under a filename reflecting which toggles were on, so different
runs don't overwrite each other and can be compared directly.
"""
import os
import random
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_THIS_DIR, os.path.dirname(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tictactoe_solver import enumerate_reachable_states, minimax
from tictactoe_coverage_eval import to_perspective, afterstate, full_coverage_accuracy
from agents.connectome_tictactoe_agent import ConnectomeTicTacToeAgent
from agents.symmetry_wrapper import SymmetryWrapper
from run_tictactoe import sample_early_boards

REPS_PER_STATE = 3
ESTIMATE_AFTER = 200
MODELS_DIR = os.path.join(_THIS_DIR, "models")

# --- toggles: change ONE at a time, see module docstring ---
SHUFFLE_STATES = True
SLEEP_EVERY_N_STATES = None      # e.g. 500 to enable; None = off
USE_SYMMETRY = True


def main():
    states = enumerate_reachable_states()
    if SHUFFLE_STATES:
        random.Random(0).shuffle(states)
    print(f"{len(states)} reachable non-terminal states to imitation-train on "
          f"({REPS_PER_STATE} reps each, 2 train() calls per rep = "
          f"~{len(states) * REPS_PER_STATE * 2} train() calls total)")
    print(f"toggles: SHUFFLE_STATES={SHUFFLE_STATES}  "
          f"SLEEP_EVERY_N_STATES={SLEEP_EVERY_N_STATES}  USE_SYMMETRY={USE_SYMMETRY}")

    calibration_boards = sample_early_boards(300)
    base_agent = ConnectomeTicTacToeAgent()
    agent = SymmetryWrapper(base_agent) if USE_SYMMETRY else base_agent
    agent.calibrate_baseline(calibration_boards)

    t0 = time.time()
    for i, (raw_board, mover) in enumerate(states):
        _, best_a = minimax(raw_board, mover)
        board_mine = to_perspective(raw_board, mover)
        legal = [a for a in range(9) if board_mine[a] == 0]
        others = [a for a in legal if a != best_a]
        if others:  # skip states with only one legal move -- no contrast to teach
            other_a = random.choice(others)
            best_after = afterstate(board_mine, best_a)
            other_after = afterstate(board_mine, other_a)
            for _ in range(REPS_PER_STATE):
                agent.update_episode([best_after], 1)
                agent.update_episode([other_after], -1)

        if SLEEP_EVERY_N_STATES and (i + 1) % SLEEP_EVERY_N_STATES == 0:
            base_agent.sleep()

        if i == ESTIMATE_AFTER:
            elapsed = time.time() - t0
            per_state = elapsed / ESTIMATE_AFTER
            total_est_min = per_state * len(states) / 60
            print(f"~{per_state:.3f}s/state -> estimated total {total_est_min:.1f} min "
                  f"for all {len(states)} states. Ctrl-C now if that's too long and "
                  f"switch to the grid-world maze instead (see module docstring).")
        if i % 500 == 0 and i > 0:
            print(f"{i}/{len(states)} states done ({time.time() - t0:.0f}s elapsed)")

    print(f"training done in {time.time() - t0:.0f}s. evaluating full-coverage accuracy...")

    acc, correct, total = full_coverage_accuracy(agent, states)
    print(f"\nfull-coverage accuracy: {correct}/{total} = {acc:.1f}%  "
          f"(baseline with all toggles off: 53.8%, untrained: 54.7%)")
    print("(this measures memorization of trained states, not generalization -- "
          "every state above WAS in the training set. Run evaluate_rl_coverage.py "
          "next for the RL-trained connectome's number on these exact same states.)")

    os.makedirs(MODELS_DIR, exist_ok=True)
    suffix = "".join([
        "_shuffle" if SHUFFLE_STATES else "",
        f"_sleep{SLEEP_EVERY_N_STATES}" if SLEEP_EVERY_N_STATES else "",
        "_symmetry" if USE_SYMMETRY else "",
    ])
    save_path = os.path.join(MODELS_DIR, f"connectome_imitation{suffix}")
    base_agent.save(save_path)
    print(f"saved to {save_path}_weights.npz / {save_path}_baseline.npy")
    if suffix:
        print("(NOT the plain 'connectome_imitation' file play_tictactoe.py looks "
              "for -- rename these two files to that if you want to play against "
              "this specific configuration)")


if __name__ == "__main__":
    main()
