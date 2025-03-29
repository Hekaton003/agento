import gymnasium as gym
from q_learning import *
import time,os

if __name__ == "__main__":
    env = gym.make('FrozenLake-v1')
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    q_table = random_q_table(-1, 0, (num_states, num_actions))
    print('Number of states:', num_states)
    print('Number of actions:', num_actions)
    print(q_table)
    num_episodes = 100
    lr = 0.01
    gamma = 0.99
    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False
        while not done:
            action = get_best_action(q_table, state)
            next_state, reward, done,truncated,info = env.step(action)
            new_q = calculate_new_q_value(q_table,state,next_state,action,reward,lr,gamma)
            q_table[state,action] = new_q
            state = next_state

    state = env.reset()[0]
    done = False
    env.render()
    while not done:
        action = get_best_action(q_table, state)
        next_state, reward, done,truncated,info = env.step(action)
        env.render()
        time.sleep(0.5)
        os.system('cls')
