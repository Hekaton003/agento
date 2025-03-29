import gymnasium as gym
import numpy as np
from mdp import value_iteration,check_policy


def check_value(V: np.ndarray):
    print('Value itteration')
    for state, value in enumerate(V[:10]):
        print(f'State: {state}, Value: {value}')


if __name__ == '__main__':
    env = gym.make('Taxi-v3', render_mode='human')
    env = env.unwrapped
    env.reset()
    env.render()
    policy, V = value_iteration(env=env, num_actions=env.action_space.n, num_states=env.observation_space.n,
                                max_episodes=50,discount_factor=0.5)
    v1 = check_policy(env,policy, 50,0.5)
    print()
    env.reset()
    env.render()
    policy, V = value_iteration(env=env, num_actions=env.action_space.n, num_states=env.observation_space.n,
                                max_episodes=50, discount_factor=0.7)
    v2 = check_policy(env, policy, 50,0.7)
    print()
    env.reset()
    env.render()
    policy, V = value_iteration(env=env, num_actions=env.action_space.n, num_states=env.observation_space.n,
                                max_episodes=50, discount_factor=0.9)
    v3 = check_policy(env,policy, 50,0.9)

    if v1 > v2 and v1 > v3:
        print(f'The best policy from value_iteration is with discount factor: 0.5 with reward:{v1}')

    elif v2 > v1 and v2 > v3:
        print(f'The best policy from value_iteration is with discount factor: 0.7 with reward: {v2}')

    else:
        print(f'The best policy from value_iteration is with discount factor: 0.9 with reward: {v3}')



