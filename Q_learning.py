import numpy as np
import matplotlib.pyplot as plt
from grid_world import grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy
from mc_exploring_starts import max_dict
from td0_prediction import random_action

GAMMA = 0.9
ALPHA = 0.1
ACTIONS = ("U", "D", "L", "R")

if __name__ == "__main__":

    grid = negative_grid(step_cost=-0.1)

    # rewards
    print("rewards")
    print_values(grid.rewards, grid)

    # no policy initialization

    # initialize Q
    Q = {}
    states = grid.all_states()
    for s in states:
        Q[s] = {}
        for a in ACTIONS:
            Q[s][a] = 0

    # now keep track of how many times Q[s] is updated
    update_counts = {}
    update_counts_sa = {}
    for s in states:
        update_counts_sa[s] = {}
        for a in ACTIONS:
            update_counts_sa[s][a] = 1.0

    # repeat until convergence
    t = 1.0
    deltas = []
    for i in range(10000):
        if i % 100 == 0:
            t += 10e-3
        if t % 2000 == 0:
            print("i", i)

        # start state
        s = (2, 0)
        grid.set_state(s)

        a, _ = max_dict(Q[s])
        biggest_change = 0
        while not grid.game_over():
            a = random_action(a, eps=0.5 / t)
            r = grid.move(a)
            s2 = grid.current_state()

            # adaptive learning rate
            alpha = ALPHA / update_counts_sa[s][a]
            update_counts_sa[s][a] += 0.005

            # we will update Q(s,a) AS we experience the episode
            old_qsa = Q[s][a]
            # the difference btwn SARSA and Q Learning is Qlearning wull use
            # max[a'] Q(s',a') in our update
            a2, max_q_s2a2 = max_dict(Q[s2])
            Q[s][a] = Q[s][a] + alpha * (r + GAMMA * max_q_s2a2 - Q[s][a])
            biggest_change = max(biggest_change, np.abs(old_qsa - Q[s][a]))

            # count how often Q[s] is updated
            update_counts[s] = update_counts.get(s, 0) + 1

            # next state becomes current state
            s = s2
            a = a2

        deltas.append(biggest_change)
    plt.plot(deltas)
    plt.show()

    # determine policy from Q* and find V* from Q*
    policy = {}
    V = {}
    for s in grid.actions.keys():
        a, max_q = max_dict(Q[s])
        policy[s] = a
        V[s] = max_q

    # what proportion of the time do we spend updating each part of Q
    print("update counts:")
    total = np.sum(list(update_counts.values()))
    for k, v in update_counts.items():
        update_counts[k] = float(v) / total
    print_values(update_counts, grid)

    print("values")
    print_values(V, grid)

    print("policy")
    print_policy(policy, grid)
