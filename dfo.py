# -*- coding: utf-8 -*-
import or_gym
from or_gym.utils import create_env
import ray
from ray.rllib import agents
from ray import tune
import ray.rllib.agents.ppo as ppo
import pickle
from scipy.optimize import minimize

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

import os
import locale
os.environ["PYTHONIOENCODING"] = "utf-8"
myLocale=locale.setlocale(category=locale.LC_ALL, locale="en_GB.UTF-8")

ray.shutdown()

def register_env(env_name, env_config={}):
    env = create_env(env_name)
    tune.register_env(env_name,
        lambda env_name: env(env_name,
            env_config=env_config))

# Environment and RL Configuration Settings
env_name = 'InvManagement-v1'
env_config = {} # Change environment parameters here


# Register environment
register_env(env_name, env_config)

# Initialize Ray and Build Agent
ray.init(ignore_reinit_error=True)

env = or_gym.make(env_name, env_config=env_config)
eps = 1000

def base_stock_policy(policy, env):
    '''
    Implements a re-order up-to policy. This means that for
    each node in the network, if the inventory at that node
    falls below the level denoted by the policy, we will
    re-order inventory to bring it to the policy level.

    For example, policy at a node is 10, current inventory
    is 5: the action is to order 5 units.
    '''
    assert len(policy) == len(env.init_inv), (
        'Policy should match number of nodes in network' +
        '({}, {}).'.format(
            len(policy), len(env.init_inv)))

    # Get echelon inventory levels
    if env.period == 0:
        inv_ech = np.cumsum(env.I[env.period] +
            env.T[env.period])
    else:
        inv_ech = np.cumsum(env.I[env.period] +
            env.T[env.period] - env.B[env.period-1, :-1])

    # Get unconstrained actions
    unc_actions = policy - inv_ech
    unc_actions = np.where(unc_actions>0, unc_actions, 0)

    # Ensure that actions can be fulfilled by checking
    # constraints
    inv_const = np.hstack([env.I[env.period, 1:], np.Inf])
    actions = np.minimum(env.c,
                np.minimum(unc_actions, inv_const))
    return actions

def dfo_func(policy, env, *args):
    '''
    Runs an episode based on current base-stock model
    settings. This allows us to use our environment for the
    DFO optimizer.
    '''
    env.reset() # Ensure env is fresh
    rewards = []
    done = False
    while not done:
        action = base_stock_policy(policy, env)
        state, reward, done, _ = env.step(action)
        rewards.append(reward)
        if done:
            break

    rewards = np.array(rewards)
    prob = env.demand_dist.pmf(env.D, **env.dist_param)

    # Return negative of expected profit
    return -1 / env.num_periods * np.sum(prob * rewards)


def optimize_inventory_policy(env_name, fun,
    init_policy=None, env_config={}, method='Powell'):

    env = or_gym.make(env_name, env_config=env_config)

    if init_policy is None:
        init_policy = np.ones(env.num_stages-1)

    # Optimize policy
    out = minimize(fun=fun, x0=init_policy, args=env,
        method=method)
    policy = out.x.copy()

    # Policy must be positive integer
    policy = np.round(np.maximum(policy, 0), 0).astype(int)

    return policy, out

policy, out = optimize_inventory_policy('InvManagement-v1',
    dfo_func)
print("Re-order levels: {}".format(policy))
print("DFO Info:\n{}".format(out))

rewards = []
for i in range(eps):
    env.reset()
    reward = 0
    while True:
        action = base_stock_policy(policy, env)
        s, r, done, _ = env.step(action)
        reward += r
        if done:
            rewards.append(reward)
            if (i+1) % 5 == 0:
                print('\rIter: {}\tReward: {:.2f}'.format(
                        i+1, reward), end='')
            break


print("Results is: ", rewards)
with open('DFO_rewards', 'wb') as fp:
    pickle.dump(rewards, fp)
ray.shutdown()
