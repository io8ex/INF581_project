# -*- coding: utf-8 -*-
import or_gym
from or_gym.utils import create_env
import ray
from ray.rllib import agents
from ray import tune
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.ars as ars
import ray.rllib.agents.a3c as a3c

import pickle
import time
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
episodes = 1000

#distributions = {1:"poisson", 2:"binom", 3:"randint", 4:"geom"}
distributions = {1:"poisson", 2:"binom", 3:"randint"}
dist_params = [
                {'mu': 20},                                                                               #Poisson : <mean value>}
                {'n': 50, 'p': 0.65},                                                                     #Binom :{'n': <mean value>, 'p': <probability between 0 and 1 of getting the mean value>}
                {'low' : 0, 'high': 300}                                                                  #Rand int  {'low' = <lower bound>, 'high': <upper bound>}
                #{'p': 0.75}                                                                               #Geom: {'p': <probability. Outcome is the number of trials to success>}
             ]

times = {"PPO" : [], "ARS" : [], "A3C": []}

for dist_type in range(2, 4):
    print("\t Analysing demand distribution type: {}".format(distributions[dist_type]))
    #Changing the demand distribution for our environment
    env_config = {"dist" : dist_type, "dist_param" : dist_params[dist_type - 1]}
    # Register environment
    register_env(env_name, env_config)

    #Initialize configurations
    rl_config_ARS = dict(
        env=env_name,
        env_config=env_config,
        framework = "tf",
        num_workers = 2,
        num_rollouts = 50,
        rollouts_used = 25,
        sgd_stepsize = 0.01,
        noise_stdev = 0.02
    )
    rl_config_A3C = dict(
        env=env_name,
        env_config=env_config,
        framework = "tf",
        num_workers = 2,
        gamma = 0.99
    )
    rl_config_PPO = dict(
        env=env_name,
        num_workers = 2,
        num_gpus = 1,
        env_config=env_config,
        model=dict(
            vf_share_layers=False,
            fcnet_activation='elu',
            fcnet_hiddens=[256, 256]
        ),
        lr=1e-5
    )

    print("------------------- PPO Agent ---------------------------------")
    begin = time.time()
    # Initialize Ray and Build Agent
    ray.init(ignore_reinit_error=True)
    agent = agents.ppo.PPOTrainer(env=env_name,
        config=rl_config_PPO)

    results = []
    for i in range(episodes):
        res = agent.train()
        results.append(res)
        if (i+1) % 5 == 0:
            print('\rIter: {}\tReward: {:.2f}'.format(i+1, res['episode_reward_mean']), end='')

    end = time.time()
    print(f"Total runtime of PPO agent is {end - begin}")
    times["PPO"].append(end - begin)
    name = "PPO_rewards_1K_eps_" + str(distributions[dist_type])
    with open(name, 'wb') as fp:
        pickle.dump(results, fp)

    # print("------------------- ARS Agent ----------------------------------")
    # begin = time.time()
    # # Initialize Ray and Build Agent
    # ray.init(ignore_reinit_error=True)
    # agent = agents.ars.ARSTrainer(env=env_name,
    #     config=rl_config_ARS)
    #
    # results = []
    # for i in range(episodes):
    #     res = agent.train()
    #     results.append(res)
    #     if (i+1) % 5 == 0:
    #         print('\rIter: {}\tReward: {:.2f}'.format(
    #                 i+1, res['episode_reward_mean']), end='')
    # end = time.time()
    # print(f"Total runtime of ARS agent is {end - begin}")
    # times["ARS"].append(end - begin)
    # name = "ARS_rewards_1K_eps_" + str(distributions[dist_type])
    # with open(name, 'wb') as fp:
    #     pickle.dump(results, fp)

    # print("------------------- A3C Agent ----------------------------------")
    # begin  = time.time()
    # # Initialize Ray and Build Agent
    # ray.init(ignore_reinit_error=True)
    # agent = agents.a3c.A3CTrainer(env=env_name,
    #     config=rl_config_A3C)
    #
    # results = []
    # for i in range(episodes):
    #     res = agent.train()
    #     results.append(res)
    #     if (i+1) % 5 == 0:
    #         print('\rIter: {}\tReward: {:.2f}'.format(
    #                 i+1, res['episode_reward_mean']), end='')
    # end = time.time()
    # print(f"Total runtime of A3C agent is {end - begin}")
    # times["A3C"].append(end - begin)
    # name = "A3C_rewards_1K_eps_" + str(distributions[dist_type])
    # with open(name, 'wb') as fp:
    #     pickle.dump(results, fp)
    print("----------------------------------------------------------------")
