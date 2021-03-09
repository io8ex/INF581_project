from environment import *
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys

from torch.autograd import Variable
from matplotlib import gridspec
import pickle
# import or_gym
from or_gym.utils import create_env
# import ray
# from ray.rllib import agents
from ray import tune

# from stable_baselines.common.env_checker import check_env

### Parameters of the model ###
ENV_NAME = 'InvManagement-v1'
env = InvManagementMasterEnv()

EPISODE_DURATION = 300

ALPHA_INIT = 0.1
SCORE = 195.0
NUM_EPISODES = 100
LEFT = 0
RIGHT = 1

VERBOSE = True

# DISCRETE_OS_SIZE = [20, 20]
# discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

### Check environment (if necessary) ###
# If the environment don't follow the interface, an error will be thrown
# check_env(env, warn=True)
# #Checking
# env.render()

### RenderWrapper for display ###
class RenderWrapper:
    def __init__(self, env, force_gif=False):
        self.env = env
        self.force_gif = force_gif
        self.reset()

    def reset(self):
        self.images = []

    def render(self):
        self.env.render()
        time.sleep(1./60.)

    def make_gif(self, filename="render"):
        if is_colab() or self.force_gif:
            imageio.mimsave(filename + '.gif', [np.array(img) for i, img in enumerate(self.images) if i%2 == 0], fps=29)
            return Image(open(filename + '.gif','rb').read())

    @classmethod
    def register(cls, env, force_gif=False):
        env.render_wrapper = cls(env, force_gif=True)

##### Policy Implementsation #####
# Constants
GAMMA = 0.9


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        super(PolicyNetwork, self).__init__()

        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.softmax(self.linear2(x), dim=1)
        return x

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(Variable(state))
        highest_prob_action = np.random.choice(self.num_actions, 3, p=np.squeeze(probs.detach().numpy()))
        # print(highest_prob_action, np.shape(np.squeeze(probs.detach().numpy())), self.num_actions)
        # highest_prob_action = np.argmax(np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob

### Update Policy ###
def update_policy(policy_network, rewards, log_probs):
    discounted_rewards = []

    for t in range(len(rewards)):
        Gt = 0
        pt = 0
        for r in rewards[t:]:
            Gt = Gt + GAMMA ** pt * r
            pt = pt + 1
        discounted_rewards.append(Gt)

    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
                discounted_rewards.std() + 1e-9)  # normalize discounted rewards

    policy_gradient = []
    for log_prob, Gt in zip(log_probs, discounted_rewards):
        policy_gradient.append(-log_prob * Gt)

    policy_network.optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    policy_network.optimizer.step()

def main(env, distr_name):
    policy_net = PolicyNetwork(env.observation_space.shape[0], 100, 128)

    max_episode_num = 1000
    max_steps = 10000
    numsteps = []
    avg_numsteps = []
    all_rewards = []
    results = []

    for episode in range(max_episode_num):
        state = env.reset()
        log_probs = []
        rewards = []
        actions = []

        for steps in range(max_steps):
            action, log_prob = policy_net.get_action(state)
            new_state, reward, done, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                total_reward = np.sum(rewards)
                update_policy(policy_net, rewards, log_probs)
                numsteps.append(steps)
                avg_numsteps.append(np.mean(numsteps[-10:]))
                all_rewards.append(total_reward)
                avg_reward = np.round(np.mean(all_rewards[-10:]), decimals=3)
                results.append(avg_reward)

                if max(all_rewards) == total_reward:
                    print(episode, total_reward)
                if episode % 50 == 0:
                    print("Number of episode: {}, Total reward: {}, Average_reward: {}, length: {}\n".format(
                        episode, np.round(np.sum(all_rewards), decimals=3), avg_reward, steps)
                    )
                break


            state = new_state

    env.close()

    ### save results #########################################
    name = "Reinforce_1001_rewards_" + distr_name
    with open(name, 'wb') as fp:
        pickle.dump(results, fp)

    ### plot results #########################################
    # p = 100
    # mean_rewards = np.array([np.mean(all_rewards[i - p:i + 1])
    #                          if i >= p else np.mean(all_rewards[:i + 1])
    #                          for i, _ in enumerate(all_rewards)])
    # std_rewards = np.array([np.std(all_rewards[i - p:i + 1])
    #                         if i >= p else np.std(all_rewards[:i + 1])
    #                         for i, _ in enumerate(all_rewards)])
    #
    #
    #
    # fig = plt.figure(constrained_layout=True, figsize=(20, 10))
    # gs = fig.add_gridspec(2, 4)
    # ax0 = fig.add_subplot(gs[:, :-2])
    # ax0.fill_between(np.arange(len(mean_rewards)),
    #                  mean_rewards - std_rewards,
    #                  mean_rewards + std_rewards,
    #                  label='Standard Deviation', alpha=0.3)
    #
    # ax0.plot(mean_rewards, label='Mean Rewards')
    # ax0.set_ylabel('Rewards')
    # ax0.set_xlabel('Episode')
    # ax0.set_title('Training Rewards')
    # ax0.legend()
    # plt.savefig('Reinforce.png')

def register_env(env_name, env_config={}):
    env = create_env(env_name)
    tune.register_env(env_name,
        lambda env_name: env(env_name,
            env_config=env_config))


if __name__ == '__main__':
    env_name = 'InvManagement-v1'
    env_config = {}  # Change environment parameters here

    distributions = {1: "poisson", 2: "binom", 3: "randint"}
    dist_params = [
        {'mu': 20},  # Poisson : <mean value>}
        {'n': 50, 'p': 0.65}, # Binom :{# 'n': <mean value>,
                                        # 'p': <probability between 0 and 1 of getting the mean value>}
        {'low': 0, 'high': 300}  # Rand int  {'low' = <lower bound>, 'high': <upper bound>}
        # {'p': 0.75}  #Geom: {'p': <probability. Outcome is the number of trials to success>}
    ]


    for dist_type in range(1, 4):
        print("\t Analysing demand distribution type: {}".format(distributions[dist_type]))
        # Changing the demand distribution for our environment
        env_config = {"dist": dist_type, "dist_param": dist_params[dist_type - 1]}
        # Register environment
        register_env(env_name, env_config)
        # Apply policy to generated environment
        main(env, distributions[dist_type])