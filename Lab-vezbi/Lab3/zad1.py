import gymnasium as gym
import torch.nn as nn
from deep_q_learning import DQN


def build_model(state_space_shape, num_actions):
    return nn.Sequential(
        nn.LazyLinear(num_actions),
        nn.Conv3d(state_space_shape, 64, num_actions),
        nn.MaxPool3d(num_actions),
        nn.Flatten(),
        nn.LazyLinear(num_actions)
    )


def get_discrete_state(state, low_value, window_size):
    new_state = (state - low_value) / window_size
    return tuple(new_state.astype(int))


if __name__ == '__main__':
    num_iterations = int(input('Select number of iterations: '))
    env = gym.make('MountainCar-v0', render_mode="rgb_array")
    env.reset()
    env.render()
    num_action = env.action_space.n
    print(num_action)
    observation_space_size = [5, 5]
    observation_space_low_value = env.observation_space.low
    observation_space_high_value = env.observation_space.high
    observation_window_size = (observation_space_high_value - observation_space_low_value) / observation_space_size
    num_episodes = 5000
    num_steps_per_episode = 100
    epsilon = 0.6
    agent = DQN(len(observation_window_size),num_action,build_model(len(observation_space_size),num_action),
                build_model(len(observation_space_size),num_action))

    for episode in range(num_episodes):
        if episode >= num_episodes * 0.4:
            epsilon = epsilon * 0.001
        state = get_discrete_state(env.reset()[0],observation_space_low_value,observation_window_size)
        for step in range(num_steps_per_episode):
            action = agent.get_action(state,epsilon)
            next_state, reward, done, _, _ = env.step(action)
            next_state = get_discrete_state(next_state, observation_space_low_value, observation_window_size)
            agent.update_memory(state, action, reward, next_state, done)
            state=next_state
        agent.train()

        if episode % 5:
            agent.update_target_model()

    print()


