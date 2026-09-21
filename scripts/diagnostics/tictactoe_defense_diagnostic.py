"""Targeted diagnostic: can each agent learn to GENERALIZE the abstract
"block an open two-in-a-row" pattern across different winning lines, or
does it only memorize the specific boards it was trained on?

Distinct from tictactoe_diagnostic.py (innate bias / cross-board
interference, connectome-only, run before the full experiment) -- this
tests ACQUISITION / representational capacity specifically, and runs
identically against all three agents (they share the same
value()/update_episode() interface), so it's a real three-way comparison,
not just a connectome check. Run this AFTER seeing win/loss/draw results
suggest a possible acquisition ceiling (e.g. near-zero draw rate against a
competent opponent even with sleep) -- it isolates whether that's a
capacity-to-represent-the-pattern problem, independent of the
retention/drift issue sleep() already partly addresses.

Setup: for each of the 8 winning lines, build a "threat" scenario -- two
opponent marks (-1) on that line, empty third cell, plus a couple of
random context marks elsewhere. Each scenario has exactly one BLOCKING
afterstate (agent plays the empty third cell) and several NON-BLOCKING
afterstates (agent plays elsewhere, leaving the line open).

TRAIN each agent on scenarios built from a subset of lines only (repeated
update_episode([block_after], +1) / update_episode([nonblock_after], -1)
calls -- the same interface real episodes use, just concentrated on this
one pattern instead of spread across everything else in a real game).

TEST on scenarios from the held-out lines (never seen during training):
does the agent now rank the blocking afterstate above the non-blocking
ones? Accuracy well above chance -> the "shared-line" concept generalizes.
Accuracy near chance -> memorizing specific boards, not the pattern.

Q-learning is included for reference ONLY -- as a tabular method with zero
generalization to unseen states, it's EXPECTED to sit near chance on
held-out lines by construction (every held-out board is a tie at 0.0),
not because it "failed" at anything. Its score is a built-in sanity check
on the test itself: if Q-learning's measured accuracy matches its own
chance level, the methodology (see the tie-breaking note below) is valid.
The informative comparison is connectome vs MLP, both of which have some
actual mechanism for generalizing across states.

IMPORTANT: candidate afterstates are shuffled and ties are broken
randomly before scoring. Without this, argmax over an all-tied (all-zero)
value list always returns whichever candidate happens to be first in a
fixed iteration order -- which would make Q-learning look like it "solved"
held-out boards purely from list-position bias, not learning.
"""
import os
import random
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_THIS_DIR, os.path.dirname(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


import numpy as np

from agents.qlearning_tictactoe_agent import QLearningTicTacToeAgent
from agents.mlp_tictactoe_agent import MLPTicTacToeAgent
from agents.connectome_tictactoe_agent import ConnectomeTicTacToeAgent
from run_tictactoe import sample_early_boards

LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
]
ROWS = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
COLS = [(0, 3, 6), (1, 4, 7), (2, 5, 8)]
DIAGS = [(0, 4, 8), (2, 4, 6)]

REPS_PER_SCENARIO = 20
N_CONTEXT_VARIANTS = 40    # bumped up from 6 -- 18 test points gave +-9.4pp noise,
                           # far too wide to read anything into a 5-10pp gap


def stratified_split(seed=0):
    """Hold out exactly one row, one column, and one diagonal -- guarantees
    training always includes at least one example of every line SHAPE, so
    a held-out test isn't accidentally a zero-shot jump to a shape the
    agent never saw at all (a plain random shuffle over 8 lines can, by
    chance, hold out both diagonals at once -- it did, the first time this
    was run)."""
    rng = random.Random(seed)
    test_lines = [rng.choice(ROWS), rng.choice(COLS), rng.choice(DIAGS)]
    train_lines = [l for l in LINES if l not in test_lines]
    return train_lines, test_lines


def scenario_seed(line, variant, salt):
    """Deterministic integer seed -- avoids Python's hash() randomization
    for strings (PYTHONHASHSEED), which would make train/test splits and
    scenario contents vary between process runs."""
    return (line[0] * 81 + line[1] * 9 + line[2]) * 1000 + variant * 10 + salt


def has_secondary_threat(board, exclude_line):
    """True if some OTHER line already has an unblocked two-in-a-row, or
    is already a completed win -- either would make "block exclude_line's
    third cell" an ambiguous or wrong label for this scenario. Random
    context marks can accidentally create this; reject and retry."""
    for ln in LINES:
        vals = [board[i] for i in ln]
        if ln == exclude_line:
            continue
        if vals.count(-1) == 2 and vals.count(0) == 1:
            return True
        if vals.count(1) == 3 or vals.count(-1) == 3:
            return True
    return False


def make_threat_scenario(line, seed, max_tries=20):
    a, b, c = line
    block_action = c
    for attempt in range(max_tries):
        rng = random.Random(seed + attempt * 7919)  # prime offset decorrelates retries
        board = [0] * 9
        board[a] = -1
        board[b] = -1
        other_empties = [i for i in range(9) if i not in line]
        rng.shuffle(other_empties)
        for i in other_empties[:2]:
            board[i] = rng.choice([-1, 1])
        if not has_secondary_threat(board, line):
            nonblock_actions = [i for i, v in enumerate(board) if v == 0 and i != block_action]
            return tuple(board), block_action, nonblock_actions
    # fallback: no context marks at all -- guaranteed clean, just less varied
    board = [0] * 9
    board[a] = -1
    board[b] = -1
    nonblock_actions = [i for i, v in enumerate(board) if v == 0 and i != block_action]
    return tuple(board), block_action, nonblock_actions


def afterstate(board, action):
    b = list(board)
    b[action] = 1
    return tuple(b)


def run_probe(agent, name):
    train_lines, test_lines = stratified_split(seed=0)  # same split for every agent

    # Build ALL training examples first, then shuffle into one randomized
    # stream before feeding them to update_episode. Feeding them
    # line-by-line (as the first version of this script did) interacts
    # badly with MLPTicTacToeAgent's replay buffer: buffer maxlen=5000 but
    # total examples here are 8000, so a sequential line-by-line order lets
    # the buffer evict most of the earliest-trained lines by the time
    # gradient steps happen late in training -- a training-curriculum
    # confound, not a capacity result. Shuffling removes it for all three
    # agents (same fairness discipline as everywhere else in this
    # project -- not singled out for MLP even though it's the one most
    # affected by it).
    training_examples = []
    for line in train_lines:
        for variant in range(N_CONTEXT_VARIANTS):
            board, block_a, nonblock_as = make_threat_scenario(line, scenario_seed(line, variant, 0))
            block_after = afterstate(board, block_a)
            nb_choice = random.Random(scenario_seed(line, variant, 1)).choice(nonblock_as)
            nonblock_after = afterstate(board, nb_choice)
            for _ in range(REPS_PER_SCENARIO):
                training_examples.append((block_after, 1))
                training_examples.append((nonblock_after, -1))
    random.Random(12345).shuffle(training_examples)
    for afterstate_board, outcome in training_examples:
        agent.update_episode([afterstate_board], outcome)

    correct, total, chance_levels = 0, 0, []
    print(f"\n--- {name}: held-out lines {test_lines} ---")
    for line in test_lines:
        for variant in range(N_CONTEXT_VARIANTS):
            board, block_a, nonblock_as = make_threat_scenario(line, scenario_seed(line, variant, 2))
            block_after = afterstate(board, block_a)
            candidates = [block_after] + [afterstate(board, a) for a in nonblock_as]

            order = list(range(len(candidates)))
            random.shuffle(order)  # kill list-position bias
            shuffled = [candidates[i] for i in order]
            values = [agent.value(s) for s in shuffled]

            best_v = max(values)
            ties = [i for i, v in enumerate(values) if v == best_v]
            picked_i = random.choice(ties)  # random tie-break, not first-wins
            picked_board = shuffled[picked_i]

            correct += int(picked_board == block_after)
            total += 1
            chance_levels.append(1.0 / len(candidates))

    acc = 100 * correct / total
    chance = 100 * float(np.mean(chance_levels))
    p = correct / total
    se = (p * (1 - p) / total) ** 0.5 * 100
    ci95 = 1.96 * se
    print(f"{name}: held-out blocking accuracy {correct}/{total} = {acc:.1f}%  "
          f"(95% CI +-{ci95:.1f}pp)  (chance ~= {chance:.1f}%)")
    return acc, chance, ci95


def main():
    calibration_boards = sample_early_boards(300)

    qlearning = QLearningTicTacToeAgent()
    mlp = MLPTicTacToeAgent()
    connectome = ConnectomeTicTacToeAgent()
    connectome.calibrate_baseline(calibration_boards)

    results = []
    for agent, name in [
        (qlearning, "qlearning (reference floor)"),
        (mlp, "mlp"),
        (connectome, "connectome"),
    ]:
        acc, chance, ci95 = run_probe(agent, name)
        results.append((name, acc, chance, ci95))

    print("\n=== summary ===")
    for name, acc, chance, ci95 in results:
        gap = acc - chance
        print(f"{name:28s} accuracy={acc:5.1f}% +-{ci95:4.1f}pp  chance={chance:5.1f}%  gap={gap:+5.1f}pp")
    print("\n-> a gap is only worth reading if it's bigger than its own CI (e.g. gap=+15pp "
          "with CI +-6pp is real; gap=+8pp with CI +-9pp is noise). qlearning's gap should "
          "be inside its own CI of 0 -- that's the sanity check on the test itself. "
          "mlp/connectome gaps clearly outside their CIs = real generalization of the "
          "'blocked line' concept. connectome's gap staying near 0 (inside its CI) next to "
          "mlp's gap being clearly positive is evidence for a representational-capacity "
          "ceiling distinct from the retention issue.")


if __name__ == "__main__":
    main()
