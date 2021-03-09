import gym
import numpy as np
from environment import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pickle
#from stable_baselines.common.env_checker import check_env

env_name = 'InvManagement-v1'
env = InvManagementMasterEnv(I0 = [100], r = [1.5, 1.0], k = [0.10, 0.075], c = [100], L = [3], h = [0.15], num_stages = 1)
#env = gym.make('FrozenLake-v0')
#env = gym.make('Taxi-v3')


# If the environment don't follow the interface, an error will be thrown
#check_env(env, warn=True)
#Cheking
#env.render()

print("Observation space: ", env.observation_space)
print("Action space: ", env.action_space)
print("Sample Action: ", env.action_space.sample())
episodes = 2000
steps = 30
alpha = 0.4
gamma = 0.999
epsilon = 0.1

#Initially, the values of the Q-table are initialized to 0. An action is chosen for a state. As we move,
#Q value is increased for the state-action whenever that action gives a good reward for the next state.
#If the action does not give a good reward for the next state, it is decreased.
#q_table = dict([(x, [1, 1, 1, 1]) for x in range(16)])

#q_table = np.zeros((env.observation_space.n, env.action_space.n))
q_table = np.zeros((env.observation_space.shape[0], env.action_space.shape[0]))
print("Q_table shape: ", q_table.shape)
def choose_action(state):
    #if the random number generated is less than 0.1, we can go for Exploration else we can go for Exploitation (Q learning).
    action = 0
    if np.random.uniform(0, 1) < epsilon:
        action.action_space.sample()
    else:
        action = np.argmax(q_table[state, :])
    return action

#SARSA depends on the current state, current action, reward obtained, next state and next action.
scores = []
all_rewards = []

for i in range(episodes):
    benefit = 0
    #env.reset: returns an array [1D * period + 3]
    #we start from the first day samplig
    state = env.reset()
    print("Initial State: ", state)
    #action = choose_action(state)
    t = 0

    for t in range(steps):
        for j in range(5):
            print("----J : {} ----- ".format(j))
            action = env.action_space.sample()
            print("Action taken: ", action)
            stateNext, reward, done, info = env.step(action)
            print("Next state: ", stateNext)

        actionNext = choose_action(stateNext)
        predict = q_table[state, action]
        target = reward + gamma * q_table[stateNext, actionNext]
        q_table[state, action] = q_table[state, action] + alpha * (target - predict)

        state, action = stateNext, actionNext
        #After one period what's the total benefit ?
        benefit += reward
        all_rewards.append(reward)

        if done:
            print("Episode {} done at {}".format(i, t))
            print("Episode {} finished after {} timesteps with r={}. Running score: {}".format(i, t, reward, benefit))
            break
    scores.append(benefit)

with open('sarsa_all_rewards', 'wb') as fp:
    pickle.dump(all_rewards, fp)

with open('sarsa_scores', 'wb') as fp:
    pickle.dump(scores, fp)

print("Scores: {}".format(scores))
