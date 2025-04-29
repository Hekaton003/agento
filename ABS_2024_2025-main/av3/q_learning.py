import numpy as np


def get_random_action(env):
    return env.action_space.sample()


def get_best_action(q_table, state):
    return np.argmax(q_table[state])


def random_q_table(lower, upper, size):
    return np.random.uniform(low=lower, high=upper, size=size)


def get_action(env, q_table, state, epsilon):
    num_actions = env.action_space.n
    probability = np.random.random() + epsilon / num_actions
    if probability < epsilon:
        return get_random_action(env)
    else:
        return get_best_action(q_table, state)


def calculate_new_q_value(q_table, old_state, new_state, action, reward, learning_rate=0.1, discount=0.99):
    max_future_q = np.max(q_table[new_state])
    if isinstance(old_state, tuple):
        current_q = q_table[old_state + (action,)]
    else:
        current_q = q_table[old_state, action]

    return (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)
