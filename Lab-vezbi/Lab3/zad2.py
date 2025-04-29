import gymnasium as gym
import torch.nn as nn
from deep_q_learning import DDQN


def eval_agent(env, agent, iterations, low, window, render=False):
    total_reward = 0
    for i in range(iterations):
        state_raw = env.reset()[0]
        state = get_discrete_state(state_raw, low, window)
        done = False
        episode_reward = 0
        while not done:
            if render:
                env.render()
            action = agent.get_action(state, 0.0)
            next_state_raw, reward, done, truncated, _ = env.step(action)
            next_state = get_discrete_state(next_state_raw, low, window)
            if done:
                episode_reward = 100
                break
            done = done or truncated
            state = next_state
            episode_reward += reward

        total_reward += episode_reward

    avg_reward = total_reward / iterations
    print(f"\n=== Evaluation Result ===")
    print(f"Average reward over {iterations} iterations: {avg_reward:.2f}")


def build_model(state_space_shape, num_actions):
    return nn.Sequential(
        nn.Linear(state_space_shape, 64),
        nn.ReLU(),
        nn.Linear(64, num_actions)
    )


def get_discrete_state(state, low_value, window_size):
    new_state = (state - low_value) / window_size
    return tuple(new_state.astype(int))


if __name__ == '__main__':
    num_iterations = int(input('Select number of iterations: '))
    env = gym.make('MountainCar-v0', render_mode="rgb_array")
    num_action = env.action_space.n
    observation_space_size = [25, 25]
    observation_space_low_value = env.observation_space.low
    observation_space_high_value = env.observation_space.high
    observation_window_size = (observation_space_high_value - observation_space_low_value) / observation_space_size
    num_episodes = 25000
    num_steps_per_episode = 100
    epsilon = 0.5
    agent = DDQN(len(observation_window_size), num_action, build_model(len(observation_space_size), num_action),
                 build_model(len(observation_space_size), num_action))

    for episode in range(num_episodes):
        if episode >= num_episodes * 0.7:
            epsilon = epsilon * 0.888
        state_raw, _ = env.reset()
        state = get_discrete_state(state_raw, observation_space_low_value, observation_window_size)
        for step in range(num_steps_per_episode):
            action = agent.get_action(state, epsilon)
            next_state_raw, reward, done, _, _ = env.step(action)
            next_state = get_discrete_state(next_state_raw, observation_space_low_value, observation_window_size)
            agent.update_memory(state, action, reward, next_state, done)
            state = next_state
        agent.train()

        if episode % 5 == 0:
            agent.update_target_model()

    eval_agent(env, agent, num_iterations, observation_space_low_value, observation_window_size)

"""
=== Evaluation Result ===
Average reward over 50 iterations: -200.00
"""
"""
=== Evaluation Result ===
Average reward over 100 iterations: -200.00
"""