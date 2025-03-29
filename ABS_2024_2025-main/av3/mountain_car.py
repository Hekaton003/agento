import os
import time

import gymnasium as gym
from q_learning import *
import numpy as np


def get_discrete_state(state, low, windows_size):
    discrete_state = (state - low) / windows_size
    return tuple(discrete_state.astype(int))


if __name__ == '__main__':
    env = gym.make('MountainCar-v0', render_mode="human")
    env.reset()
    env.render()
    discrete_state_size = [20, 20]
    state_high_value = env.observation_space.high
    state_low_value = env.observation_space.low
    num_actions = env.action_space.n
    staten_windows_size = (state_high_value - state_low_value) / discrete_state_size
    q_table = random_q_table(-1, 0, (discrete_state_size + [num_actions]))
    num_episodes = 200
    lr = 0.1
    gamma = 0.99
    for episode in range(num_episodes):
        state = get_discrete_state(env.reset()[0], state_low_value, staten_windows_size)
        done = False
        while not done:
            action = get_best_action(q_table, state)
            next_state, reward, done, truncated, info = env.step(action)
            next_state = get_discrete_state(next_state, state_low_value, staten_windows_size)
            new_q = calculate_new_q_value(q_table, state, next_state, action, reward, lr, gamma)
            q_table[state + (action,)] = new_q
            state = next_state

    state = get_discrete_state(env.reset()[0], state_low_value, staten_windows_size)
    done = False
    env.render()
    while not done:
        action = get_best_action(q_table, state)
        next_state, reward, done, truncated, info = env.step(action)
        next_state = get_discrete_state(next_state, state_low_value, staten_windows_size)
        env.render()
        time.sleep(0.5)
        os.system('cls')
