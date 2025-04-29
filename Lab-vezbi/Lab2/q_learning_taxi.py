import gymnasium as gym
from q_learning import get_random_action,get_best_action, get_action, \
    random_q_table, calculate_new_q_value


def check_policy(env, q_table, num_iterations: int, discount: float,lr:float):
    total_average_reward = 0
    total_average_steps = 0
    for i in range(num_iterations):
        state = env.reset()[0]
        done = False
        total_reward = 0
        counter = 0
        env.render()
        while not done:
            counter += 1
            action = get_best_action(q_table, state)
            next_state, reward, done, truncated, info = env.step(action)
            env.render()
            total_reward += reward
            state = next_state
            if done:
                total_average_reward += (total_reward / counter)
                total_average_steps += counter
                break
            elif truncated:
                break

    print(f'Overall average reward is: {total_average_reward / num_iterations} from {num_iterations} iterations with '
          f'discount factor {discount},learning rate {lr}')
    print(f'Overall average steps is: {total_average_steps / num_iterations} from {num_iterations} iterations with '
          f'discount factor {discount}, learning rate {lr}')


if __name__ == '__main__':
    with_epsilon = int(input('Select 1 - Yes or 0- No for epsilon policy: '))
    learning_rate = float(input('Select learning rate: '))
    discount = float(input('Select discount factor: '))
    num_iterations = int(input('Select number of iterations: '))
    env = gym.make('Taxi-v3', render_mode='ansi')

    num_states = env.observation_space.n
    num_actions = env.action_space.n

    q_table = random_q_table(-1, 0, (num_states, num_actions))

    num_episodes = 200
    num_steps_per_episode = 5

    for episode in range(num_episodes):
        state = env.reset()[0]
        for step in range(num_steps_per_episode):
            if with_epsilon:
                action = get_action(env, q_table, state,0.5)
            else:
                action = get_best_action(q_table, state)

            new_state, reward, terminated, _, _ = env.step(action)

            new_q = calculate_new_q_value(q_table,
                                          state, new_state,
                                          action, reward,
                                          learning_rate, discount)

            q_table[state, action] = new_q

            state = new_state

            print()

    check_policy(env,q_table,num_iterations,discount,learning_rate)

# Ако наместо најдобрата акција се избира акција на случаен начин при учење на Q табелата тогаш резултатите не се менуваат
"""
'Select 1 - Yes or 0- No for epsilon policy: 0
Select learning rate: 0.1
Select discount factor: 0.9
Select number of iterations: 50

Se dobiva:
Overall average reward is: 0.0 from 50 iterations with discount factor 0.9,learning rate 0.1
Overall average steps is: 0.0 from 50 iterations with discount factor 0.9, learning rate 0.1

So epsilon se dobiva:
 Overall average reward is: 0.0 from 50 iterations with discount factor 0.9,learning rate 0.1
 Overall average steps is: 0.0 from 50 iterations with discount factor 0.9, learning rate 0.1
"""

"""
'Select 1 - Yes or 0- No for epsilon policy: 0
 Select learning rate: 0.01
 Select discount factor: 0.9
 Select number of iterations: 50

 Se dobiva:
 Overall average reward is: 0.0 from 50 iterations with discount factor 0.9,learning rate 0.01
 Overall average steps is: 0.0 from 50 iterations with discount factor 0.9, learning rate 0.01
 So epsilon se dobiva:
 Overall average reward is: 0.0 from 50 iterations with discount factor 0.9,learning rate 0.01
 Overall average steps is: 0.0 from 50 iterations with discount factor 0.9, learning rate 0.01
"""

"""
'Select 1 - Yes or 0- No for epsilon policy: 0
 Select learning rate: 0.1
 Select discount factor: 0.5
 Select number of iterations: 50

 Se dobiva:
 Overall average reward is: 0.0 from 50 iterations with discount factor 0.5,learning rate 0.1
 Overall average steps is: 0.0 from 50 iterations with discount factor 0.5, learning rate 0.1
 So epsilon se dobiva:
 Overall average reward is: 0.0 from 50 iterations with discount factor 0.5,learning rate 0.1
 Overall average steps is: 0.0 from 50 iterations with discount factor 0.5, learning rate 0.1
"""

"""
'Select 1 - Yes or 0- No for epsilon policy: 0
 Select learning rate: 0.01
 Select discount factor: 0.5
 Select number of iterations: 50

 Se dobiva:
 Overall average reward is: 0.0 from 50 iterations with discount factor 0.5,learning rate 0.01
 Overall average steps is: 0.0 from 50 iterations with discount factor 0.5, learning rate 0.01
 So epsilon se dobiva:
 Overall average reward is: 0.0 from 50 iterations with discount factor 0.5,learning rate 0.01
 Overall average steps is: 0.0 from 50 iterations with discount factor 0.5, learning rate 0.01
"""

"""
'Select 1 - Yes or 0- No for epsilon policy: 0
Select learning rate: 0.1
Select discount factor: 0.9
Select number of iterations: 100

Se dobiva:
Overall average reward is: 0.0 from 100 iterations with discount factor 0.9,learning rate 0.1
Overall average steps is: 0.0 from 100 iterations with discount factor 0.9, learning rate 0.1
So epsilon se dobiva:
Overall average reward is: 0.0 from 100 iterations with discount factor 0.9,learning rate 0.1
Overall average steps is: 0.0 from 100 iterations with discount factor 0.9, learning rate 0.1
"""

"""
'Select 1 - Yes or 0- No for epsilon policy: 0
 Select learning rate: 0.01
 Select discount factor: 0.9
 Select number of iterations: 100
 Se dobiva:
 Overall average reward is: 0.0 from 100 iterations with discount factor 0.9,learning rate 0.01
 Overall average steps is: 0.0 from 100 iterations with discount factor 0.9, learning rate 0.01
 So epsilon se dobiva:
 Overall average reward is: 0.0 from 100 iterations with discount factor 0.9,learning rate 0.01
 Overall average steps is: 0.0 from 100 iterations with discount factor 0.9, learning rate 0.01
"""

"""
'Select 1 - Yes or 0- No for epsilon policy: 0
 Select learning rate: 0.1
 Select discount factor: 0.5
 Select number of iterations: 100
 Se dobiva:
 Overall average reward is: 0.0 from 100 iterations with discount factor 0.5,learning rate 0.1
 Overall average steps is: 0.0 from 100 iterations with discount factor 0.5, learning rate 0.1
 So epsilon se dobiva:
 Overall average reward is: 0.0 from 100 iterations with discount factor 0.5,learning rate 0.1
 Overall average steps is: 0.0 from 100 iterations with discount factor 0.5, learning rate 0.1
"""

"""
'Select 1 - Yes or 0- No for epsilon policy: 0
 Select learning rate: 0.01
 Select discount factor: 0.5
 Select number of iterations: 100
 Se dobiva:
 Overall average reward is: 0.0 from 100 iterations with discount factor 0.5,learning rate 0.01
 Overall average steps is: 0.0 from 100 iterations with discount factor 0.5, learning rate 0.01
 So epsilon se dobiva:
 Overall average reward is: 0.0 from 100 iterations with discount factor 0.5,learning rate 0.01
 Overall average steps is: 0.0 from 100 iterations with discount factor 0.5, learning rate 0.01
"""