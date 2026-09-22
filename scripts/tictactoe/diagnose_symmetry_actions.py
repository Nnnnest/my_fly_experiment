# scripts/tictactoe/diagnose_symmetry_actions.py
import itertools
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tictactoe_symmetry import canonicalize, action_to_canonical, action_from_canonical, _NAMES

def test_roundtrip():
    """Every action, under every transform, must round-trip exactly."""
    bad = []
    for name in _NAMES:
        for a in range(9):
            back = action_from_canonical(action_to_canonical(a, name), name)
            if back != a:
                bad.append((name, a, action_to_canonical(a, name), back))
    if bad:
        print(f"FAIL: {len(bad)} round-trip mismatches:")
        for row in bad:
            print("  transform=%s a=%d -> canon=%d -> back=%d" % row)
    else:
        print("OK: action round-trip is exact for all 9 actions x 8 transforms")

def test_full_episode_consistency():
    """For every reachable board, canonicalize + map every legal action to
    canonical space and back; the mapped-back set must equal the original
    legal set exactly (same elements, no duplicates, no drops)."""
    from tictactoe_solver import enumerate_reachable_states
    states = enumerate_reachable_states()
    bad = 0
    for board, mover in states:
        board_mine = tuple(mover * v for v in board)
        legal = [a for a in range(9) if board_mine[a] == 0]
        canon, name = canonicalize(board_mine)
        canon_legal = [action_to_canonical(a, name) for a in legal]
        back = sorted(action_from_canonical(ca, name) for ca in canon_legal)
        if back != sorted(legal):
            bad += 1
            if bad <= 5:
                print(f"MISMATCH board={board_mine} legal={legal} -> canon_legal={canon_legal} -> back={back}")
    print(f"{'FAIL' if bad else 'OK'}: {bad}/{len(states)} states had a legal-set mismatch")

if __name__ == "__main__":
    test_roundtrip()
    test_full_episode_consistency()
