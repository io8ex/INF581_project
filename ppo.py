# -*- coding: utf-8 -*-
import or_gym
from or_gym.utils import create_env
import ray
from ray.rllib import agents
from ray import tune
import ray.rllib.agents.ppo as ppo
import pickle

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
#config = ppo.DEFAULT_CONFIG.copy()
#config["num_gpus"] = 1
#config["num_workers"] = 2
#config["framework"] = "tf"
#config["log_level"] = "WARN"

rl_config = dict(
    env=env_name,
    num_workers = 3,
    num_gpus = 1,
    env_config=env_config,
    model=dict(
        vf_share_layers=False,
        fcnet_activation='elu',
        fcnet_hiddens=[256, 256]
    ),
    lr=1e-5
)


# Register environment
register_env(env_name, env_config)

# Initialize Ray and Build Agent
ray.init(ignore_reinit_error=True)
agent = agents.ppo.PPOTrainer(env=env_name,
    config=rl_config)

#print("Dashboard URL: {}".format(ray.get_webui_url()))

results = []
for i in range(500):
    res = agent.train()
    results.append(res)
    if (i+1) % 5 == 0:
        print('\rIter: {}\tReward: {:.2f}'.format(
                i+1, res['episode_reward_mean']), end='')


print("Results is: ", results)
with open('PPO_rewards', 'wb') as fp:
    pickle.dump(results, fp)
ray.shutdown()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Unpack values from each iteration
rewards = np.hstack([i['hist_stats']['episode_reward']
    for i in results])
pol_loss = [
    i['info']['learner']['default_policy']['policy_loss']
    for i in results]
vf_loss = [
    i['info']['learner']['default_policy']['vf_loss']
    for i in results]

p = 100
mean_rewards = np.array([np.mean(rewards[i-p:i+1])
                if i >= p else np.mean(rewards[:i+1])
                for i, _ in enumerate(rewards)])
std_rewards = np.array([np.std(rewards[i-p:i+1])
               if i >= p else np.std(rewards[:i+1])
               for i, _ in enumerate(rewards)])

fig = plt.figure(constrained_layout=True, figsize=(20, 10))
gs = fig.add_gridspec(2, 4)
ax0 = fig.add_subplot(gs[:, :-2])
ax0.fill_between(np.arange(len(mean_rewards)),
                 mean_rewards - std_rewards,
                 mean_rewards + std_rewards,
                 label='Standard Deviation', alpha=0.3)

ax0.plot(mean_rewards, label='Mean Rewards')
ax0.set_ylabel('Rewards')
ax0.set_xlabel('Episode')
ax0.set_title('Training Rewards')
ax0.legend()

ax1 = fig.add_subplot(gs[0, 2:])
ax1.plot(pol_loss)
ax1.set_ylabel('Loss')
ax1.set_xlabel('Iteration')
ax1.set_title('Policy Loss')

ax2 = fig.add_subplot(gs[1, 2:])
ax2.plot(vf_loss)
ax2.set_ylabel('Loss')
ax2.set_xlabel('Iteration')
ax2.set_title('Value Function Loss')

plt.savefig("PPO Results.jpg")
plt.show()
