import random


def argmax_random(items, values, tol=1e-9):
    """Argmax with uniformly random tie-breaking. Consumes no random numbers when there is no tie."""
    top = max(values)
    best = [it for it, v in zip(items, values) if v >= top - tol]
    return best[0] if len(best) == 1 else random.choice(best)
