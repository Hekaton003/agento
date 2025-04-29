import gymnasium as gym
from mdp import policy_iteration,check_policy


if __name__ == '__main__':
    discount = float(input('Select discount: '))
    num_iterations = int(input('Select number of iterations: '))
    env = gym.make('Taxi-v3', render_mode='ansi')
    env = env.unwrapped
    env.reset()
    env.render()
    policy, V = policy_iteration(env=env, num_actions=env.action_space.n, num_states=env.observation_space.n,
                                 max_iteration=50, discount_factor=0.5)

    print('Policy iteration')
    check_policy(env,policy, num_iterations, discount)

#Kaj policy iteration prosecnite nagradi se pogolemi od onie so value iteration

"""
'
Select discount: 0.5
Select number of iterations: 50
Policy iteration
Overall average reward is: 0.7022285410667765 from 50 iterations with discount factor 0.5
Overall average steps is: 12.94 from 50 iterations with discount factor 0.5

"""

"""
'
Select discount: 0.7
Select number of iterations: 50
Policy iteration
Overall average reward is: 0.7474700397641574 from 50 iterations with discount factor 0.7
Overall average steps is: 12.62 from 50 iterations with discount factor 0.7
"""

"""
'
Select discount: 0.9
Select number of iterations: 50
Policy iteration
Overall average reward is: 0.7152456807897982 from 50 iterations with discount factor 0.9
Overall average steps is: 13.0 from 50 iterations with discount factor 0.9
"""

"""
'
Select discount: 0.5
Select number of iterations: 100
Policy iteration
Overall average reward is: 0.6970909262306318 from 100 iterations with discount factor 0.5
Overall average steps is: 13.03 from 100 iterations with discount factor 0.5
"""

"""
'
Select discount: 0.7
Select number of iterations: 100
Policy iteration
Overall average reward is: 0.7361137734814204 from 100 iterations with discount factor 0.7
Overall average steps is: 12.84 from 100 iterations with discount factor 0.7
"""

"""
'
Select discount: 0.9
Select number of iterations: 100
Policy iteration
Overall average reward is: 0.7214532771150415 from 100 iterations with discount factor 0.9
Overall average steps is: 12.9 from 100 iterations with discount factor 0.9
"""


