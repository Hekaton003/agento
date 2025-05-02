import gymnasium as gym
from deep_q_learning_torch import DDPG, OrnsteinUhlenbeckActionNoise


def eval_agent(env, agent, iterations, render=False):
    total_reward = 0
    for i in range(iterations):
        state_raw = env.reset()[0]
        done = False
        episode_reward = 0
        while not done:
            if render:
                env.render()
            action = agent.get_action(state_raw, discrete=False)
            next_state, reward, terminated,truncated, _ = env.step(action)
            state_raw = next_state
            done = terminated or truncated
            episode_reward += reward

        total_reward += episode_reward

    avg_reward = total_reward / iterations
    print(f"\n=== Evaluation Result ===")
    print(f"Average reward over {iterations} iterations: {avg_reward:.2f}")


if __name__ == '__main__':
    num_iterations = int(input('Select number of iterations: '))
    env = gym.make('LunarLanderContinuous-v2', render_mode='rgb_array')
    env.reset()

    agent = DDPG(state_space_shape=(8,), action_space_shape=(2,),
                 learning_rate_actor=0.001, learning_rate_critic=0.001,
                 discount_factor=0.9, batch_size=64, memory_size=1000)

    num_episodes = 200
    num_steps_per_episode = 1000
    epsilon = 0.6
    noise = OrnsteinUhlenbeckActionNoise(action_space_shape=(2,))

    for episode in range(num_episodes):
        state, _ = env.reset()
        if episode >= 0.6 * num_episodes:
            epsilon = epsilon * 0.86
        for step in range(num_steps_per_episode):
            if episode % 5 == 0:
                action = agent.get_action(state,epsilon, discrete=False) + noise()
            else:
                action = agent.get_action(state,epsilon, discrete=False)
            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update_memory(state, action, reward, new_state, terminated)
            state = new_state
            if done:
                break
        if (episode + 1) % 5 == 0:
            agent.train()

        if (episode + 1) % 20 == 0:
            agent.update_target_model()

    eval_agent(env,agent,num_iterations)

"""
=== Evaluation Result ===
Average reward over 50 iterations: -255.56
"""

"""
=== Evaluation Result ===
Average reward over 100 iterations: -267.04
"""