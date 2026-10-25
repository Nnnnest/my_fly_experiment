"""Operand-pair-disjoint sampling for the addition task: a fraction of the (a, b) pairs is held out for evaluation
and never used for training or calibration, so no evaluation triple (a, b, c) occurs in any training stream."""
import numpy as np


def split_pairs(op_max, seed, held_frac=0.2):
    pairs = [(a, b) for a in range(op_max + 1) for b in range(op_max + 1)]
    perm = np.random.default_rng([seed, 9]).permutation(len(pairs))
    n_held = max(1, int(round(held_frac * len(pairs))))
    held = [pairs[i] for i in perm[:n_held]]
    train = [pairs[i] for i in perm[n_held:]]
    return train, held


def triple_for(rng, a, b, op_max, positive, two_digit):
    """Same label and negative-sampling rules as agents.shared.addition_common.sample_triple, for fixed (a, b)."""
    correct = a + b
    c_max = 2 * op_max
    if positive:
        return a, b, correct, True
    r = rng.random()
    if two_digit and r < 0.3:
        ct, cu = correct // 10, correct % 10
        c = cu * 10 + ct
    elif r < 0.7:
        c = min(max(correct + int(rng.choice([-3, -2, -1, 1, 2, 3])), 0), c_max)
    else:
        c = int(rng.integers(0, c_max + 1))
    if c == correct:
        c = correct + 1 if correct < c_max else correct - 1
    return a, b, c, False


def draw(rng, pairs, n, op_max, two_digit, pos_frac=0.5):
    out = []
    for _ in range(n):
        a, b = pairs[int(rng.integers(len(pairs)))]
        out.append(triple_for(rng, a, b, op_max, bool(rng.random() < pos_frac), two_digit))
    return out
