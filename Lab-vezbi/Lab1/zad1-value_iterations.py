import gymnasium as gym
import numpy as np
from mdp import value_iteration,check_policy


def check_value(V: np.ndarray):
    print('Value itteration')
    for state, value in enumerate(V[:10]):
        print(f'State: {state}, Value: {value}')


if __name__ == '__main__':
    discount = float(input('Select discount: '))
    num_iterations = int(input('Select number of iterations: '))
    env = gym.make('Taxi-v3', render_mode='ansi')
    env = env.unwrapped
    env.reset()
    env.render()
    policy, V = value_iteration(env=env, num_actions=env.action_space.n, num_states=env.observation_space.n,
                                max_episodes=50,discount_factor=0.5)

    print('Value Iteration')
    check_policy(env,policy, num_iterations,discount)

"""
'
Select discount: 0.5
Select number of iterations: 50
Value Iteration
Overall average reward is: 0.6208749143013849 from 50 iterations with discount factor 0.5
Overall average steps is: 13.6 from 50 iterations with discount factor 0.5
"""

"""
'
Select discount: 0.7
Select number of iterations: 50
Value Iteration
Overall average reward is: 0.6198440970793913 from 50 iterations with discount factor 0.7
Overall average steps is: 13.44 from 50 iterations with discount factor 0.7
"""

"""
'
Select discount: 0.9
Select number of iterations: 50
Value Iteration
Overall average reward is: 0.6320292403674757 from 50 iterations with discount factor 0.9
Overall average steps is: 13.32 from 50 iterations with discount factor 0.9
"""

"""
'
Select discount: 0.5
Select number of iterations: 100
Value Iteration
Overall average reward is: 0.6565595776772244 from 100 iterations with discount factor 0.5
Overall average steps is: 13.19 from 100 iterations with discount factor 0.5
"""

"""
'
Select discount: 0.7
Select number of iterations: 100
Value Iteration
Overall average reward is: 0.6833094062799943 from 100 iterations with discount factor 0.7
Overall average steps is: 13.01 from 100 iterations with discount factor 0.7
"""

"""
'
Select discount: 0.9
Select number of iterations: 100
Value Iteration
Overall average reward is: 0.6862510969422732 from 100 iterations with discount factor 0.9
Overall average steps is: 12.92 from 100 iterations with discount factor 0.9
"""

