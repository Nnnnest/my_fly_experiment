"""Shared full-coverage evaluation: for every reachable non-terminal
board, does the agent pick a minimax-optimal move? Used by both
train_imitation_tictactoe.py (evaluating the imitation-trained
connectome) and evaluate_rl_coverage.py (evaluating an RL-trained
connectome on the EXACT same states), so the two numbers are directly
comparable.

"Full coverage" means every state below either WAS available to
imitation training, or COULD have come up during RL self-play -- this
measures competence on the full reachable state space, not generalization
to unseen states (that's what tictactoe_defense_diagnostic.py tests).
"""
import random

from tictactoe_solver import enumerate_reachable_states, minimax


def to_perspective(raw_board, mover):
    return tuple(mover * v for v in raw_board)


def afterstate(board, action):
    b = list(board)
    b[action] = 1
    return tuple(b)


def full_coverage_accuracy(agent, states=None):
    """agent needs best_action(board, legal_actions) (preferred -- matches
    what real play uses) or falls back to value(afterstate). Returns
    (accuracy_pct, correct, total)."""
    if states is None:
        states = enumerate_reachable_states()

    correct = 0
    for raw_board, mover in states:
        best_val, _ = minimax(raw_board, mover)
        board_mine = to_perspective(raw_board, mover)
        legal = [a for a in range(9) if board_mine[a] == 0]
        if len(legal) == 1:
            correct += 1
            continue

        if hasattr(agent, "best_action"):
            picked_a = agent.best_action(board_mine, legal)
        else:
            candidates = [afterstate(board_mine, a) for a in legal]
            values = [agent.value(s) for s in candidates]
            top_v = max(values)
            ties = [i for i, v in enumerate(values) if v == top_v]
            picked_a = legal[random.choice(ties)]

        # "correct" = picked move is ALSO optimal (minimax often has
        # several tied-best moves), not just matches minimax's one
        # canonical choice
        new_raw = list(raw_board)
        new_raw[picked_a] = mover
        v2, _ = minimax(tuple(new_raw), -mover)
        picked_val = -v2
        correct += int(picked_val == best_val)

    total = len(states)
    return 100 * correct / total, correct, total
