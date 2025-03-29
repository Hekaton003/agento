import gymnasium as gym
from q_learning import random_q_table, get_action, calculate_new_q_value, get_best_action


def check_policy(env,q_table, num_iterations: int, num_episodes: int,low:float,window_size:float):
    total_average_reward = 0
    total_average_steps = 0
    for i in range(num_iterations):
        print(f'Iteration: {i+1}')
        state = get_discrete_state(env.reset()[0],low,window_size)
        done = False
        total_reward = 0
        counter = 0
        env.render()
        while not done:
            counter += 1
            action = get_best_action(q_table, state)
            next_state, reward, done, truncated, info = env.step(action)
            next_state = get_discrete_state(next_state,low,window_size)
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
          f'episodes {num_episodes}')
    print(f'Overall average steps is: {total_average_steps / num_iterations} from {num_iterations} iterations with '
          f'episodes {num_episodes}')


def get_discrete_state(state, low_value, window_size):
    new_state = (state - low_value) / window_size
    return tuple(new_state.astype(int))


if __name__ == '__main__':
    with_decay = int(input('With decay 1 or 0: '))
    decay_value = 1
    if with_decay:
        decay_value = float(input('Enter decay value: '))
    num_iterations = int(input('Select number of iterations: '))
    num_episodes = int(input('Select number of episodes: '))
    env = gym.make('MountainCar-v0',render_mode="rgb_array")

    num_actions = env.action_space.n

    observation_space_size = [3, 3]
    observation_space_low_value = env.observation_space.low
    observation_space_high_value = env.observation_space.high
    observation_window_size = (observation_space_high_value - observation_space_low_value) / observation_space_size

    q_table = random_q_table(-1, 0, (observation_space_size + [num_actions]))
    lr = 0.1
    gamma = 0.99
    epsilon = 0.6
    for episode in range(num_episodes):
        print(f'Episode: {episode}')
        if episode >= num_iterations * 0.4:
            epsilon = epsilon * decay_value

        state = get_discrete_state(env.reset()[0], observation_space_low_value, observation_window_size)
        done = False
        while not done:
            action = get_action(env, q_table, state, epsilon)
            next_state, reward, done, truncated, info = env.step(action)
            next_state = get_discrete_state(next_state, observation_space_low_value, observation_window_size)
            new_q = calculate_new_q_value(q_table, state, next_state, action, reward, lr, gamma)
            q_table[state + (action,)] = new_q
            state = next_state

    check_policy(env,q_table,num_iterations,num_episodes,observation_space_low_value,observation_window_size)

"""
'With decay 1 or 0: 0 
 Select number of iterations: 100
 Select number of episodes: 200
 Se dobiva:
 Overall average reward is: 0.0 from 100 iterations with episodes 200
 Overall average steps is: 0.0 from 100 iterations with episodes 200
 So decay ke se dobie:
 Overall average reward is: 0.0 from 100 iterations with episodes 200
  Overall average steps is: 0.0 from 100 iterations with episodes 200
"""
"""
'With decay 1 or 0: 0 
 Select number of iterations: 50
 Select number of episodes: 200
 Se dobiva:
 Overall average reward is: 0.0 from 50 iterations with episodes 200
 Overall average steps is: 0.0 from 50 iterations with episodes 200
 So decay ke se dobie:
 Overall average reward is: 0.0 from 50 iterations with episodes 200
 Overall average steps is: 0.0 from 50 iterations with episodes 200
"""
