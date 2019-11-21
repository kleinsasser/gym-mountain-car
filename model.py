import gym
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim

env = gym.make('MountainCar-v0')
print(env)
print('action space: ', env.action_space)
print('observation space: ', env.observation_space)

#hyperparameters
gamma = 0.99
lr = 0.01
optim_freq = 25
render = True

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(2, 64)
        self.dropout = nn.Dropout(0.5)
        self.affine2 = nn.Linear(64, 3)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=lr)
eps = np.finfo(np.float32).eps.item()

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()

def optimize():
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = R * gamma + r
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    loss = torch.cat(policy_loss).sum()
    optimizer.step()
    del policy.saved_log_probs[:]
    del policy.rewards[:]

state = env.reset()
step_reward = 0
for i in range(10000):
    if render: env.render()
    action = select_action(state)
    state, reward, done, _ = env.step(action)
    policy.rewards.append(reward)
    step_reward += reward

    if i % optim_freq == 0:
        optimize()
        print('Step Reward: ', step_reward)
        step_reward = 0

env.close()