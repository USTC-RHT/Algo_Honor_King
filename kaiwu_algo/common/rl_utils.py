import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

# 用于 noisy_dqn中的策略评估
def evaluate_policy(env, agent, turns = 3):
    agent.Q_main.eval() # Take deterministic actions at test time
    total_scores = 0
    for _ in range(turns):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_scores += reward
            state = next_state
    agent.Q_main.train()
    return int(total_scores/turns)

# 用于 ppo中的策略评估
def evaluate_policy_ppo(env, agent, turns = 3):
    agent.actor.eval() # Take deterministic actions at test time
    total_scores = 0
    for _ in range(turns):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.best_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_scores += reward
            state = next_state
    agent.actor.train()
    return int(total_scores/turns)

# 用于 noisy_dqn中的神经网络构造
class NoisyLinear(nn.Module):
    '''From https://github.com/Lizhi-sjtu/DRL-code-pytorch/blob/main/3.Rainbow_DQN/network.py'''
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters() # for mu and sigma
        self.reset_noise() # for epsilon

    def forward(self, x):
        ''' Sample a noisy network '''
        if self.training:
            self.reset_noise()
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)  # mul是对应元素相乘
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_i = self.scale_noise(self.in_features)
        epsilon_j = self.scale_noise(self.out_features)
        self.weight_epsilon.copy_(torch.ger(epsilon_j, epsilon_i))
        self.bias_epsilon.copy_(epsilon_j)

    def scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x