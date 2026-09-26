"""Procedural circle-counting generator for the shape-counting experiment
(pivot from digit shape-identity, see shapes_diagnostic.py results: even
a maximally-distinct digit subset {0,1,7,8} still showed 0.77-0.84 KC
overlap -- this connectome configuration does not separate fine
shape-identity/positional distinctions, so this task deliberately tests a
GRADED magnitude judgment (how many blobs / how much ink) instead of a
discrete conjunctive pattern, matching the architecture's demonstrated
strength (bandit: single graded stimulus->value, worked) rather than its
demonstrated weakness (tic-tac-toe / digit-identity: discrete
positional/conjunctive patterns, both failed).

n non-overlapping filled circles at random positions on a size x size
canvas. Total "ink" scales with n by construction (same radius per
circle), which is intentional -- it gives the connectome a genuine
intensity/magnitude cue to exploit, rather than requiring it to literally
enumerate discrete objects the way a symbolic counter would.
"""
import numpy as np


def circle_bitmap(n, rng, size=32, radius=3, max_tries=300):
    """n non-overlapping filled circles, deterministic given rng's state."""
    if n == 0:
        return np.zeros((size, size), dtype=np.float32)
    centers = []
    min_dist = 2 * radius + 1
    for i in range(n):
        for _ in range(max_tries):
            cx = rng.uniform(radius, size - radius)
            cy = rng.uniform(radius, size - radius)
            if all(((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5 >= min_dist
                   for ox, oy in centers):
                centers.append((cx, cy))
                break
        else:
            raise RuntimeError(
                f"couldn't place circle {i}/{n} without overlap in a {size}x{size} "
                f"canvas at radius={radius}; increase size, reduce radius, or lower n_max")
    yy, xx = np.mgrid[0:size, 0:size]
    canvas = np.zeros((size, size), dtype=np.float32)
    for cx, cy in centers:
        canvas = np.maximum(canvas, (((xx - cx) ** 2 + (yy - cy) ** 2) <= radius ** 2).astype(np.float32))
    return canvas


def sample(n, rng, size=32, radius=3, noise_std=0.0):
    base = circle_bitmap(n, rng, size=size, radius=radius)
    if noise_std <= 0:
        return base
    noisy = base + rng.normal(0.0, noise_std, size=base.shape)
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def sample_random(rng, n_min=1, n_max=4, size=32, radius=3, noise_std=0.0):
    """(count, bitmap) for a uniformly random count in [n_min, n_max]."""
    n = int(rng.integers(n_min, n_max + 1))
    return n, sample(n, rng, size=size, radius=radius, noise_std=noise_std)


if __name__ == "__main__":
    # sanity check: confirm every count 0..N_MAX actually fits at the
    # chosen size/radius before spending full-mode compute on it
    N_MAX = 4
    rng = np.random.default_rng(0)
    for n in range(N_MAX + 1):
        bm = circle_bitmap(n, rng, size=32, radius=3)
        print(f"n={n}: {int(bm.sum())} lit pixels")
