"""Procedural 5-digit (0-4) shape generator for the shape-association
experiment (summary.txt's proposed option (c): show the connectome an
image, teach it to name the digit).

Seven-segment-style strokes on a small (7x5) grid, upscaled via
nearest-neighbor resize to `size`x`size` (default 16x16). No external
dataset -- fully reproducible from a seed, and cheap to regenerate at a
different resolution or digit count if the current settings need
changing after the baseline run.

Digits 0-4 only for now (keeps mode='full' runtime manageable at ~1s/step
per fly.step() call -- see fly_api.py's README limits section). Extend
SEGMENTS to 5-9 later if the 5-digit set works.
"""
import numpy as np

_SEG_H, _SEG_W = 7, 5

# standard seven-segment layout: a=top, b=upper-right, c=lower-right,
# d=bottom, e=lower-left, f=upper-left, g=middle
SEGMENTS = {
    0: "abcdef",
    1: "bc",
    2: "abged",
    3: "abgcd",
    4: "fgbc",
    5: "afgcd",
    6: "afgedc",
    7: "abc",
    8: "abcdefg",
    9: "abcdfg",
}

# Chosen for minimal pairwise shared-segment count (maximally distinct
# strokes), as opposed to the sequential 0-4 set which happens to include
# 2/3 (differ by exactly 1 segment) -- use this subset (or pick your own
# via --digits) to test whether digit-SET CHOICE, not resolution/hops, was
# the real bottleneck behind the 2/3/4 KC-overlap cluster.
DISTINCT_SUBSET = (0, 1, 7, 8)


def _draw_segment(g, seg):
    if seg == "a":
        g[0, 1:4] = 1.0
    elif seg == "b":
        g[1:3, 4] = 1.0
    elif seg == "c":
        g[4:6, 4] = 1.0
    elif seg == "d":
        g[6, 1:4] = 1.0
    elif seg == "e":
        g[4:6, 0] = 1.0
    elif seg == "f":
        g[1:3, 0] = 1.0
    elif seg == "g":
        g[3, 1:4] = 1.0
    else:
        raise ValueError(f"unknown segment '{seg}'")


def _seg_grid(digit):
    g = np.zeros((_SEG_H, _SEG_W), dtype=np.float32)
    for seg in SEGMENTS[digit]:
        _draw_segment(g, seg)
    return g


def _resize(g, size):
    gh, gw = g.shape
    rows = np.arange(size) * gh // size
    cols = np.arange(size) * gw // size
    return g[np.ix_(rows, cols)]


def digit_bitmap(digit, size=16):
    """Clean (noise-free) bitmap for `digit`, size x size, values in [0,1]."""
    if digit not in SEGMENTS:
        raise ValueError(f"digit {digit} not in {sorted(SEGMENTS)}")
    return _resize(_seg_grid(digit), size)


DIGIT_SHAPES = {d: digit_bitmap(d) for d in SEGMENTS}


def sample(label, rng, size=16, noise_std=0.0):
    """One bitmap for `label`. noise_std=0 (default) for training; pass
    noise_std>0 only for the held-out generalization check (same
    held-out-condition philosophy as tictactoe_defense_diagnostic.py --
    train clean, test noisy, never train on the noisy version)."""
    base = digit_bitmap(label, size=size)
    if noise_std <= 0:
        return base.copy()
    noisy = base + rng.normal(0.0, noise_std, size=base.shape)
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def sample_random(rng, n_digits=5, size=16, noise_std=0.0):
    """(label, bitmap) for a uniformly random digit in range(n_digits)."""
    label = int(rng.integers(n_digits))
    return label, sample(label, rng, size=size, noise_std=noise_std)


if __name__ == "__main__":
    # quick sanity print: confirm the 5 shapes are pairwise distinct
    for d, bm in DIGIT_SHAPES.items():
        print(f"digit {d}: {int(bm.sum())} lit pixels")
    flat = np.stack([DIGIT_SHAPES[d].reshape(-1) for d in sorted(DIGIT_SHAPES)])
    dists = ((flat[:, None, :] - flat[None, :, :]) ** 2).sum(-1)
    print("pairwise L2^2 distances (0 on the diagonal, all others should be >0):")
    print(dists.astype(int))
