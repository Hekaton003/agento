import gymnasium as gym
from mdp import policy_iteration,check_policy


if __name__ == '__main__':
    env = gym.make('Taxi-v3', render_mode='human')
    env = env.unwrapped
    env.reset()
    env.render()
    policy, V = policy_iteration(env=env, num_actions=env.action_space.n, num_states=env.observation_space.n,
                                 max_iteration=50, discount_factor=0.5)
    p1 = check_policy(env,policy, 50, 0.5)
    print()
    env.reset()
    env.render()
    policy, V = policy_iteration(env=env, num_actions=env.action_space.n, num_states=env.observation_space.n,
                                 max_iteration=50, discount_factor=0.7)
    p2 = check_policy(env,policy, 50, 0.7)
    print()
    env.reset()
    env.render()
    policy, V = policy_iteration(env=env, num_actions=env.action_space.n, num_states=env.observation_space.n,
                                 max_iteration=50, discount_factor=0.9)
    p3 = check_policy(env,policy, 50, 0.9)

    if p1 > p2 and p1 > p3:
        print(f'The best policy from policy_iteration is with discount factor: 0.5 with reward: {p1}')

    elif p2 > p1 and p2 > p3:
        print(f'The best policy from policy_iteration is with discount factor: 0.7 with reward: {p2}')

    else:
        print(f'The best policy from policy_iteration is with discount factor: 0.9 with reward: {p3}')
