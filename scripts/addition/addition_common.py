"""Shared sampling/eval helpers for all three addition-experiment tiers,
both single-digit and two-digit."""
import numpy as np


def sample_triple(rng, op_max, positive, two_digit=False):
    a = int(rng.integers(0, op_max + 1))
    b = int(rng.integers(0, op_max + 1))
    correct = a + b
    c_max = 2 * op_max
    if positive:
        return a, b, correct, True
    r = rng.random()
    if two_digit and r < 0.3:
        ct, cu = correct // 10, correct % 10
        c = cu * 10 + ct  # digit-swap distractor (wrong carry)
    elif r < 0.7:
        delta = int(rng.choice([-3, -2, -1, 1, 2, 3]))
        c = min(max(correct + delta, 0), c_max)
    else:
        c = int(rng.integers(0, c_max + 1))
    if c == correct:
        c = correct + 1 if correct < c_max else correct - 1
    return a, b, c, False


def sample_batch(rng, op_max, n, two_digit=False, pos_frac=0.5):
    return [sample_triple(rng, op_max, rng.random() < pos_frac, two_digit) for _ in range(n)]


def accuracy(agent, triples):
    hits = sum(agent.predict(a, b, c) == label for a, b, c, label in triples)
    return hits / max(1, len(triples))
