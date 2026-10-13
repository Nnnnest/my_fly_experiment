from envs.tictactoe_env import TicTacToe


def play(policies):
    """policies: {+1: fn(board_mine, legal)->action, -1: ...}. Returns (afterstates per mark, result per mark)
    where result is +1 win / 0 draw / -1 loss from that mark's perspective."""
    env = TicTacToe()
    env.reset()
    after = {1: [], -1: []}
    while True:
        mark = env.to_move
        a = policies[mark](env.board_from_perspective(mark), env.legal_actions())
        _, outcome, done = env.step(a)
        after[mark].append(env.board_from_perspective(mark))
        if done:
            return after, {mark: 1 if outcome == 1 else 0, -mark: -1 if outcome == 1 else 0}


def train_policy(agent, eps):
    return lambda b, legal: agent.choose_action(b, legal, eps)


def greedy_policy(agent):
    return lambda b, legal: agent.best_action(b, legal)


def opp_policy(fn, mark):
    return lambda b, legal: fn(b, legal, mark)
