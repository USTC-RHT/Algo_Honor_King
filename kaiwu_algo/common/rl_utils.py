import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math
import copy

# 正交初始化
def orthogonal_init(layer, gain=1.0):
    '''增益(gain) 控制正交矩阵的幅度'''
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)
            
def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

#reward engineering for better training
def reward_shaping(reward, env_name):
    if env_name == 'Pendulum-v1':
        reward = (reward + 8) / 8

    elif env_name in ['LunarLander-v3','LunarLanderContinuous-v3']:
        if reward <= -100: reward = -10

    elif env_name in ['BipedalWalker-v3','BipedalWalkerHardcore-v3']:
        if reward <= -100: reward = -1
    return reward

# 用于 prioritized_dqn 与 noisy_dqn中的策略评估
def evaluate_policy_dqn(env, agent, turns = 3):
    agent.Q_main.eval() # Take deterministic actions at test time
    total_scores = 0
    for _ in range(turns):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.predict(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_scores += reward
            state = next_state
    agent.Q_main.train()
    return int(total_scores/turns)

# 用于 ppo_discrete、ddpg中的策略评估
def evaluate_policy_ppo_discrete_ddpg(env, agent, turns = 3):
    agent.actor.eval() # Take deterministic actions at test time
    total_scores = 0
    for _ in range(turns):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.exploit(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_scores += reward
            state = next_state
    agent.actor.train()
    return int(total_scores/turns)

# 用于 ppo_continuous中的策略评估
def evaluate_policy_ppo_continuous(env, agent, turns = 3):
    agent.actor.eval() # Take deterministic actions at test time
    total_scores = 0
    max_action = float(env.action_space.high[0])
    for _ in range(turns):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.exploit(state)
            env_action = 2 * (action - 0.5) * max_action
            next_state, reward, terminated, truncated, _ = env.step(env_action)
            done = terminated or truncated
            total_scores += reward
            state = next_state
    agent.actor.train()
    return int(total_scores/turns)

# 用于 MADDPG 中的策略评估
'''episode_limit 表示评估时每一个episode的最大步数'''
def evaluate_policy_maddpg(env, agent_num, agent_n, turns = 3, episode_limit = 25):
    total_score = [0] * agent_num
    for _ in range(turns):
        obs_n = env.reset()
        episode_reward = [0] * agent_num
        for _ in range(episode_limit):
            a_n = [agent_n[i].exploit(obs_n[i]).astype(np.float32) for i in range(agent_num)]
            obs_next_n, reward_n, dw_n, _ = env.step(copy.deepcopy(a_n))
            for i in range(agent_num):
                episode_reward[i] += reward_n[i]
            obs_n = obs_next_n
            done = any(dw_n)
            if done: break
        total_score = [total_score[i] + episode_reward[i] for i in range(agent_num)]
    return [score / turns for score in total_score]


# 用于 MAPPO_SMAC 中的策略评估
def evaluate_policy_mappo(env, use_rnn, agent_n, episode_limit, turns = 32):
    win_times = 0
    evaluate_reward = 0
    for _ in range(turns):    
        win_tag = False
        episode_reward = 0
        env.reset()
        if use_rnn:
            agent_n.actor.rnn_hidden = None
            agent_n.critic.rnn_hidden = None
        for _ in range(episode_limit):
            obs_n = env.get_obs()  # obs_n.shape=(N,obs_dim)
            avail_a_n = env.get_avail_actions()  # avail_a_n.shape=(N,action_dim)
            a_n = agent_n.exploit(obs_n,avail_a_n)

            r, done, info = env.step(a_n)
            win_tag = True if done and 'battle_won' in info and info['battle_won'] else False
            episode_reward += r
            if done: break
        if win_tag: win_times += 1
        evaluate_reward += episode_reward

    win_rate = win_times / turns
    evaluate_reward = evaluate_reward / turns
    return win_rate, evaluate_reward

# 用于 VDN_SMAC与 QMIX_SMAC 中的策略评估
def evaluate_policy_vdn_qmix(env, config, use_rnn, agent_n, episode_limit, turns = 32):
    win_times = 0
    evaluate_reward = 0
    for _ in range(turns):    
        win_tag = False
        episode_reward = 0
        env.reset()
        if use_rnn:
            agent_n.Q_main.rnn_hidden = None
        last_onehot_a_n = np.zeros((config.agent_num, config.action_dim_n[0]))
        for _ in range(episode_limit):
            obs_n = env.get_obs()  # obs_n.shape=(N,obs_dim)
            avail_a_n = env.get_avail_actions()  # avail_a_n.shape=(N,action_dim)
            a_n = agent_n.exploit(obs_n,avail_a_n,last_onehot_a_n)
            last_onehot_a_n = np.eye(config.action_dim_n[0])[a_n]
            r, done, info = env.step(a_n)
            win_tag = True if done and 'battle_won' in info and info['battle_won'] else False
            episode_reward += r
            if done: break
        if win_tag: win_times += 1
        evaluate_reward += episode_reward

    win_rate = win_times / turns
    evaluate_reward = evaluate_reward / turns
    return win_rate, evaluate_reward

# 用于 MAPPO_SMAC 中的奖励归一化
class Normalization:
    """Normalize data using running mean and standard deviation."""
    def __init__(self, shape):
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def learn(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)

    def normalize(self, x):
        self.learn(x)
        return (x - self.mean) / (self.std + 1e-8)

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

'''利用 Gumbel-Softmax 的方法来得到离散分布的近似采样 用于MADDPG算法'''
def onehot_from_logits(logits, eps=0.01):
    # 获取最大值索引并生成one-hot
    argmax_actions = torch.zeros_like(logits).scatter(1, logits.argmax(dim=1, keepdim=True), 1.0)

    # 随机选择动作，并转换成one-hot形式
    random_actions = torch.eye(logits.shape[1], device=logits.device)[torch.randint(0, logits.shape[1], (logits.shape[0],))]

    # epsilon-贪婪选择
    return torch.where(torch.rand(logits.shape[0], device=logits.device) > eps, argmax_actions, random_actions)


def sample_gumbel(shape, delta=1e-20):
    """ 从Gumbel(0,1)分布中采样 """
    U = torch.rand(shape, dtype=torch.float32)
    return -torch.log(-torch.log(U + delta) + delta)


def gumbel_softmax_sample(logits, temperature):
    """ 从Gumbel-Softmax分布中采样 """
    y = logits + sample_gumbel(logits.shape).to(logits.device)
    return F.softmax(y / temperature, dim=1)


def gumbel_softmax(logits, temperature=1.0):
    """从Gumbel-Softmax分布中采样,并进行离散化"""
    y_soft = gumbel_softmax_sample(logits, temperature)
    y_hard = onehot_from_logits(y_soft)

    '''直通估计器(Straight-Through Estimator,STE    ):
    前向传播时使用y_hard, 反向传播时使用y_soft的梯度'''
    y = y_soft + (y_hard - y_soft).detach()
    return y