import gymnasium as gym
import numpy as np
import torch.nn as nn
from PIL import Image
from deep_q_learning_torch import DQN


def eval_agent(env, agent, iterations, render=False):
    total_reward = 0
    for i in range(iterations):
        state, _ = env.reset()
        preprocessed_state = preprocess_state(state)
        terminated = False
        episode_reward = 0
        while not terminated:
            if render:
                env.render()
            action = agent.get_action(preprocessed_state, 0.0)
            next_state, reward, terminated,truncated,info = env.step(action)
            new_preprocessed_state = preprocess_state(new_state)
            episode_reward += reward
            preprocessed_state = new_preprocessed_state

        total_reward += episode_reward

    avg_reward = total_reward / iterations
    print(f"\n=== Evaluation Result ===")
    print(f"Average reward over {iterations} iterations: {avg_reward:.2f}")


def build_model(input_channels, num_actions):
    return nn.Sequential(
        nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.LazyLinear(num_actions)
    )


def preprocess_state(state):
    img = Image.fromarray(state)
    img = img.convert('L')
    grayscale_img = np.array(img, dtype=np.float32)
    grayscale_img = grayscale_img / 255.0
    return grayscale_img[np.newaxis, :, :]


def preprocess_reward(reward):
    return np.clip(reward, -1000.0, 1000.0)


if __name__ == '__main__':
    num_iterations = int(input('Select number of iterations: '))
    env = gym.make('ALE/MsPacman-v5', render_mode="rgb_array")
    state, _ = env.reset()
    state_space_shape = env.observation_space.shape
    num_actions = env.action_space.n

    agent = DQN(state_space_shape=state_space_shape, num_actions=num_actions,model=build_model(1,num_actions),
                target_model=build_model(1,num_actions))

    num_episodes = 100
    epsilon = 0.5

    for episode in range(num_episodes):
        if episode >= num_episodes * 0.6:
            epsilon = epsilon * 0.06
        preprocessed_state = preprocess_state(state)
        terminated = False
        while not terminated:
            action = agent.get_action(preprocessed_state, epsilon)
            new_state, reward, terminated, _, _ = env.step(action)
            new_preprocessed_state = preprocess_state(new_state)
            agent.update_memory(preprocessed_state, action, reward, new_preprocessed_state, terminated)

            preprocessed_state=new_preprocessed_state

        if (episode + 1) % 5 == 0:
            agent.train()

        if (episode + 1) % 20 == 0:
            agent.update_target_model()

    eval_agent(env,agent,num_iterations)

"""
=== Evaluation Result ===
Average reward over 50 iterations: 70.00
"""
"""
=== Evaluation Result ===
Average reward over 100 iterations: 210.00
"""