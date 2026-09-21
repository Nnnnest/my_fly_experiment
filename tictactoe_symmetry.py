"""3x3 board dihedral-symmetry canonicalization: every board has up to 8
equivalent orientations (4 rotations x mirror) that are game-theoretically
identical -- a rotated or mirrored board has exactly the same optimal
moves, just relabeled. canonicalize() picks one consistent representative
orientation for any board, so "visually different but equivalent" boards
collapse onto the exact same encoded pattern for an agent, instead of
competing separately for capacity/weights.

Reachable non-terminal tic-tac-toe states: ~4,520 raw, ~765 canonical --
roughly an 8x reduction in how many genuinely distinct patterns an agent
needs to represent.

Uses numpy's rot90/fliplr/flipud/transpose directly (rather than
hand-derived permutation tables) so the transforms are correct by
construction, not by manual derivation.
"""
import numpy as np


def _to_grid(board):
    return np.array(board, dtype=int).reshape(3, 3)


def _to_flat(grid):
    return tuple(int(x) for x in grid.flatten())


_TRANSFORMS = {
    "identity": lambda g: g,
    "rot90": lambda g: np.rot90(g, k=1),
    "rot180": lambda g: np.rot90(g, k=2),
    "rot270": lambda g: np.rot90(g, k=3),
    "flip_h": lambda g: np.fliplr(g),
    "flip_v": lambda g: np.flipud(g),
    "flip_diag": lambda g: g.T,
    "flip_antidiag": lambda g: np.rot90(g.T, k=2),
}
_INVERSE_NAME = {
    "identity": "identity", "rot90": "rot270", "rot180": "rot180",
    "rot270": "rot90", "flip_h": "flip_h", "flip_v": "flip_v",
    "flip_diag": "flip_diag", "flip_antidiag": "flip_antidiag",
}
_NAMES = list(_TRANSFORMS.keys())


def apply_transform(board, name):
    return _to_flat(_TRANSFORMS[name](_to_grid(board)))


def canonicalize(board):
    """Returns (canonical_board, transform_name). transform_name is the
    transform that maps the ORIGINAL board to the canonical orientation --
    use it (via action_to_canonical/action_from_canonical) to translate a
    single action index between the two orientations. Ties (a genuinely
    symmetric board matches its own canonical form under >1 transform) are
    broken by _NAMES order -- doesn't matter which, the canonical board
    itself is identical either way."""
    variants = [(apply_transform(board, name), name) for name in _NAMES]
    variants.sort(key=lambda vn: vn[0])
    return variants[0]


def _transform_index(i, name):
    marker = [0] * 9
    marker[i] = 1
    return apply_transform(tuple(marker), name).index(1)


def action_to_canonical(action, transform_name):
    """Original-orientation action index -> canonical-orientation index."""
    return _transform_index(action, transform_name)


def action_from_canonical(canonical_action, transform_name):
    """Canonical-orientation action index -> original-orientation index."""
    return _transform_index(canonical_action, _INVERSE_NAME[transform_name])


if __name__ == "__main__":
    # sanity check: canonicalizing all 8 orientations of the same board
    # should produce the same canonical board every time
    board = (1, -1, 0, 0, 1, 0, -1, 0, 0)
    canon0, _ = canonicalize(board)
    for name in _NAMES:
        rotated = apply_transform(board, name)
        canon, _ = canonicalize(rotated)
        assert canon == canon0, f"mismatch under {name}: {canon} != {canon0}"
    print("all 8 orientations canonicalize identically -- OK")
