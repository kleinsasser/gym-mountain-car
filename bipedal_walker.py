import gym
import Box2D
import mujoco_py

import numpy as np
from random import randint
from statistics import mean

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim

env = gym.make('BipedalWalker-v2')

print(env.action_space)
print(env.observation_space)

class Policy(nn.Module):

    def __init__(self, rng_seeds):
        super(Policy, self).__init__()

        self.rng_seeds = rng_seeds
        torch.manual_seed(rng_seeds[0][0])

        self.affine1 = nn.Linear(24, 24, bias=False)
        self.affine2 = nn.Linear(24, 4, bias=False)

        self.drop = nn.Dropout(0.5)

        for seed, sigma in self.rng_seeds[1:len(self.rng_seeds)]:
            torch.manual_seed(seed)
            for name, tensor in self.named_parameters():
                to_add = torch.normal(0.0, sigma, tensor.size())
                self.drop(to_add)
                tensor.data.add_(to_add)

    def forward(self, x):
        x = self.affine1(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

def select_action(policy, state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    return probs.detach().numpy()[0]

def random_seed():
    return randint(0, 2 ** 31 - 1)


######################################################################
######################################################################

policies = []
policy_rewards = []
state = env.reset()
sigma = 0.5
max_sigma = 0.5
min_sigma = 0.01
render = False

episode_diffs = []
prev_reward = -1000
regressions = 0
prev_best = -1000

first = True
for episode in range(10000):
    if first:
        # initial generation, all random policies
        for _ in range(200):
            policy = Policy([(random_seed(), sigma)])
            running_reward = 0
            stopped_frames = 0
            while True:
                action = select_action(policy, state)
                state, reward, done, _ = env.step(action)
                running_reward += reward

                if abs(state[2]) < 0.03:
                    stopped_frames += 1
                else:
                    stopped_frames = 0
                if stopped_frames >= 100:
                    done = True
                    running_reward -= 100

                if done:
                    policies.append(policy)
                    policy_rewards.append(running_reward)
                    state = env.reset()
                    break
        first = False
    else:
        # get best policy from previous policies
        i_of_best = policy_rewards.index(max(policy_rewards))
        print('Best reward of generation', episode, ': ', max(policy_rewards))
        print('Sigma: ', sigma)

        if max(policy_rewards) >= -30:
            sigma = 0.1
            render = True

        if max(policy_rewards) < -85:
            regressions += 1
        else:
            regressions = 0

        if regressions >= 10:
            sigma = min([max_sigma, sigma * 2])

        best = policies[i_of_best]
        ep_first = True

        policies = []
        policy_rewards = []

        for _ in range(25):
            if ep_first:
                policy = best
            else:
                seeds = best.rng_seeds.copy()
                seeds.append((random_seed(), sigma))
                policy = Policy(seeds)

            running_reward = 0
            stopped_frames = 0
            while (True):
                if ep_first and render: env.render()
                action = select_action(policy, state)
                state, reward, done, _ = env.step(action)
                running_reward += reward

                if abs(state[2]) < 0.03:
                    stopped_frames += 1
                else:
                    stopped_frames = 0
                if stopped_frames >= 100:
                    done = True
                    running_reward -= 100

                if done:
                    policies.append(policy)
                    policy_rewards.append(running_reward)
                    env.reset()
                    if ep_first:
                        ep_first = False
                    break
env.close()