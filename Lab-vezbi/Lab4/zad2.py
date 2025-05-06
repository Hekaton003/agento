import gymnasium as gym
from stable_baselines3 import DDPG


def eval_agent(env, agent, iterations, render=False):
    total_reward = 0
    for i in range(iterations):
        state_raw = env.reset()[0]
        done = False
        episode_reward = 0
        while not done:
            if render:
                env.render()
            action,_ = agent.predict(state_raw,deterministic=False)

            next_state, reward, terminated, truncated, info = env.step(action)
            state_raw = next_state
            done = terminated
            episode_reward += reward

        total_reward += episode_reward

    avg_reward = total_reward / iterations
    print(f"\n=== Evaluation Result ===")
    print(f"Average reward over {iterations} iterations: {avg_reward:.2f}")


if __name__ == '__main__':
    num_iterations = int(input('Select number of iterations: '))
    env = gym.make('LunarLanderContinuous-v2', render_mode='rgb_array')
    model = DDPG(policy='MlpPolicy', env=env, learning_rate=0.004, buffer_size=1000, batch_size=16,
                 train_freq=(5, 'episode'),
                 device='cpu', verbose=1, gamma=0.7)

    model.learn(total_timesteps=10000, log_interval=10)
    eval_agent(env,model,num_iterations)

"""
=== Evaluation Result ===
Average reward over 50 iterations: -354.97
"""
"""
=== Evaluation Result ===
Average reward over 100 iterations: -589.47
"""