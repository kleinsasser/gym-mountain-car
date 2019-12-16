import gym
import numpy as np
from random import randint

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim

env = gym.make('MountainCar-v0')
render = True
np.random.seed(0)

class Policy(nn.Module):

    def __init__(self, rng_seeds):
        super(Policy, self).__init__()

        self.rng_seeds = rng_seeds
        torch.manual_seed(rng_seeds[0][0])

        self.affine1 = nn.Linear(2, 32, bias=False)
        self.affine2 = nn.Linear(32, 3, bias=False)

        for seed, sigma in self.rng_seeds[1:len(self.rng_seeds)]:
            torch.manual_seed(seed)
            for name, tensor in self.named_parameters():
                to_add = torch.normal(0.0, sigma, tensor.size())
                tensor.data.add_(to_add)

    def forward(self, x):
        x = self.affine1(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

def select_action(policy, state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    return action.item()

def random_seed():
    return randint(0, 2 ** 31 - 1)

policies = []
policy_rewards = []
state = env.reset()
sigma = 0.4

first = True
for episode in range(200):
    if first:
        for _ in range(25):
            policy = Policy([(random_seed(), sigma)])
            best_dist = -10
            while True:
                action = select_action(policy, state)
                state, reward, done, _ = env.step(action)
                if state[1] > best_dist: best_dist = state[1]
                if done:
                    policies.append(policy)
                    policy_rewards.append(best_dist)
                    state = env.reset()
                    break
        first = False
    else:
        # get best policy from previous policies)
        i_of_best = policy_rewards.index(max(policy_rewards))
        print('Best reward of generation', episode, ': ', max(policy_rewards))

        complete = 0
        for i in policy_rewards:
            if i >= 0.06:
                complete += 1
        ratio = complete/len(policy_rewards)
        print('Percent of generation succeeded: ', ratio)
        if ratio > 0.0: sigma = 0.2
        if ratio > 0.2: sigma = 0.1
        if ratio > 0.4: sigma = 0.05
        if ratio > 0.6: sigma = 0.025
        if ratio > 0.8: sigma = 0.0125

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
            best_dist = -10
            while (True):
                if ep_first: env.render()
                action = select_action(policy, state)
                state, reward, done, _ = env.step(action)
                if state[1] > best_dist: best_dist = state[1]
                if done:
                    policies.append(policy)
                    policy_rewards.append(best_dist)
                    env.reset()
                    if ep_first:
                        ep_first = False
                    break
env.close()