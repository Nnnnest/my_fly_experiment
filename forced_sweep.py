import numpy as np

def forced_exploration_sweep(env, agent, min_pulls, n_passes=1, seed=0):
    """One full pass = every traversable state, every action, once.
    Unlike the old Bellman-bootstrap version, this update rule trains each
    (state, action) pair on a fixed, local fact (did this move closer to the
    goal?) that doesn't depend on any other pair's current value — so there's
    no propagation to wait for, and n_passes=1 (with min_pulls repeats per
    pair) is normally enough. Raise it only if agent._corrected_values()
    still looks unstable after inspection.
    """
    rng = np.random.default_rng(seed)
    states = env.traversable_states()
    pairs = [(pos, a) for pos in states for a in range(4)]

    total_updates = min_pulls * n_passes
    for _ in range(total_updates):
        rng.shuffle(pairs)
        for pos, action in pairs:
            state_id = env.state_id(pos)
            env.set_pos(pos)
            s2, reward, done = env.step(action)
            agent.update(state_id, action, reward, s2, done)
