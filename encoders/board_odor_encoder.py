import numpy as np

N_CELLS = 9
live_alpn = np.load("live_alpn_indices.npy")
GROUP_SIZE = len(live_alpn) // N_CELLS  # ~33 at hops=1
CELL_GROUPS = [live_alpn[i*GROUP_SIZE:(i+1)*GROUP_SIZE] for i in range(N_CELLS)]

# calibrate this empirically in the diagnostic script below
CELL_DRIVE = 1.0

def board_to_vector(board, n_alpn=685):
    """board: length-9 tuple from board_from_perspective (+1 self, -1 opp, 0 empty)."""
    vec = np.zeros(n_alpn)
    for cell_idx, mark in enumerate(board):
        if mark != 0:
            vec[CELL_GROUPS[cell_idx]] = mark * CELL_DRIVE
    return vec
