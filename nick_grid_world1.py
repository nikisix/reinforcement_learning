import math
import random

import numpy as np
import matplotlib.pyplot as plt
from grid_world import grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy

debug = False
thresh = 1e-4
SMALL_ENOUGH = thresh
EPSILON = 0.9  # Greed-level
GAMMA = 0.9
ACTIONS = ("U", "D", "L", "R")
ALL_POSSIBLE_ACTIONS = ACTIONS


def initilize():
    # we use the negative grid so we can make the agent as efficient as possible
    grid = negative_grid()

    # print rewards
    print("rewards")
    print_values(grid.rewards, grid)

    # state -> action
    # well randomly choose an action and update as we learn
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

    # initial policy
    print("initial random policy")
    print_policy(policy, grid)

    # initialize V(state)
    V = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            V[s] = np.random.random()
        else:
            # terminal state
            V[s] = 0
    return V, policy, grid


def find_best_move(grid, s, V):
    """Returns the best action and reward given a grid state. Tried is an optimization only
    """
    best_action = ''
    best_reward = float('-inf')
    r = grid.rewards[s]  # current reward
    r_prime = float('-inf')  # reward at next time-step
    for action in grid.actions[s]:
        r_prime = grid.move(action)
        s = grid.current_state()
        if best_reward < r_prime:
            best_reward = r_prime
            best_action = action
        grid.undo_move(action)
    V[s] = math.tanh(V[s] + GAMMA * best_reward)
    return best_action, best_reward, V


if __name__ == "__main__":
    V, policy, grid = initilize()
    tried = list()  #state-action tuples
    last_value = float('-inf')
    max_iters = 2000
    num_iters = 0
    cum_reward = 0

    while(True):
        num_iters += 1
        if max_iters < num_iters: break
        s = grid.current_state()  # aesthetic
        last_value = V[s]

        if np.random.uniform() < EPSILON:  # Going with the best choice
            best_action, best_reward, V = find_best_move(grid, s, V)

            if (V[s] - best_reward) < thresh:  # Unless the reward isn't any better than our
                # current valuation of this square
                selected_action = np.random.choice(grid.actions[s])
            else:
                policy[s] = best_action
                selected_action = best_action

        else:  # Explore: pick a random action
            selected_action = np.random.choice(grid.actions[s])

        if debug: print(s, selected_action)

        reward = grid.move(selected_action)
        cum_reward += reward
        while grid.is_terminal(grid.current_state()):
            print(f'resetting from terminal state')
            grid.set_state(random.choice(list(grid.all_states())))


    print("final values")
    print_values(V, grid)
    print()

    print("final policy")
    print_policy(policy, grid)
    print(f"total number of iterations {num_iters}")
    print(f"total reward: {cum_reward}")
