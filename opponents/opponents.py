import random

def random_opponent(legal_actions, board=None):
    return random.choice(legal_actions)

def heuristic_opponent(board, legal_actions, me):
    """Win if possible, else block, else center, else corner, else random."""
    opp = -me
    for a in legal_actions:
        b = list(board); b[a] = me
        if winner_of(b) == me:
            return a
    for a in legal_actions:
        b = list(board); b[a] = opp
        if winner_of(b) == opp:
            return a
    if 4 in legal_actions:
        return 4
    corners = [c for c in (0, 2, 6, 8) if c in legal_actions]
    if corners:
        return random.choice(corners)
    return random.choice(legal_actions)
