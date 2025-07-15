import numpy as np
import collections
import random
import copy
import torch
import torch.nn as nn
from sac_continuous_algo import Algo
from sac_continuous_config import Config
from dataclasses import dataclass
from torch.distributions import Normal
import torch.nn.functional as F

'''s, a, r, s_next, dw'''
@dataclass
class SampleData:
    state: float
    action: int
    reward: float
    next_state: float
    dw: bool

# ReplayBuffer
class ReplayBuffer():
    
    ''' 经验回放池 
        Max number of transitions to store in the buffer. 
        When the buffer overflows the old memories are dropped.
    '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, dw):  # 将数据加入buffer
        sample_data = SampleData(state, action, reward, next_state, dw)
        self.buffer.append(sample_data)
        return

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        return transitions

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)

# Soft策略网络构造
class Actor(torch.nn.Module):
    def __init__(self, state_dim, hid_shape, action_dim):
        super().__init__()
        layers = []
        layer_shape = [state_dim] + list(hid_shape)
        '''设置激活函数为 ReLU '''
        activation = nn.ReLU
        '''Build networks with For loop'''
        for j in range(len(layer_shape)-1):
            layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), activation()]
        # 用Sequential包装共享层
        self.shared_net = nn.Sequential(*layers)
        
        '''分头: mu和log_sigma分别是两个线性层'''
        self.mu_head = nn.Linear(layer_shape[-1], action_dim)
        self.log_sigma_head = nn.Linear(layer_shape[-1], action_dim)
        self.log_sigma_min, self.log_sigma_max = Config.LOG_SIGMA_RANGE
     
    def forward(self, x):
        x = self.shared_net(x)
        mu = self.mu_head(x)
        log_sigma = self.log_sigma_head(x)
        '''learn log_std rather than std, so that exp(log_std) is always > 0'''
        log_sigma = torch.clamp(log_sigma,self.log_sigma_min,self.log_sigma_max)
        sigma = torch.exp(log_sigma)
        '''仅返回均值与方差'''
        return mu, sigma
    
    def dist(self,x):
        mu, sigma = self.forward(x)
        '''高斯分布'''
        return Normal(mu,sigma)
    
    def enforce_action_bounds(self,u,dist):
        '''↓↓↓ Enforcing Action Bounds ↓↓↓'''
        a = torch.tanh(u)
        ''' Get probability density of logp_pi_a from probability density of u:
        logp_pi_a = (dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=1, keepdim=True)
        Derive from the above equation. No a, thus no tanh(h), thus less gradient vanish and more stable.'''
        logp_pi_a = dist.log_prob(u).sum(axis=1, keepdim=True) \
        - (2 * (torch.log(torch.tensor(2.0)) - u - F.softplus(-2 * u))).sum(axis=1, keepdim=True)
        return a, logp_pi_a
    
    def sample_act(self,state):
        dist = self.dist(state)
        '''rsample() 通过重参数化技巧(reparameterization trick),让采样过程可微,从而可以用梯度下降法优化参数'''
        u = dist.rsample()
        a, logp_pi_a = self.enforce_action_bounds(u,dist)
        return a, logp_pi_a
    
    def deterministic_act(self,state):
        mu, _ = self.forward(state)
        return mu

    def transform_sample_data(self, list_sample_data, device):
        State = []
        for sample_data in list_sample_data:
            State.append(sample_data.state)
            
        tensor_state = torch.tensor(np.array(State)).to(device)
        return tensor_state

class MLP_QNet(nn.Module):
    def __init__(self, state_dim, hid_shape, action_dim):
        super().__init__()
        layers = []
        layer_shape = [state_dim + action_dim] + list(hid_shape) + [1]
        '''设置激活函数为 ReLU '''
        activation = nn.ReLU
        '''Build networks with For loop'''
        for j in range(len(layer_shape)-1):
            if j < len(layer_shape) - 2: 
                layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), activation()]
            else: 
                layers += [nn.Linear(layer_shape[j], layer_shape[j+1])]
        self.Q = nn.Sequential(*layers)  

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.Q(sa)

# Soft价值网络构造
class Double_Q_Critic(nn.Module):
    def __init__(self, state_dim, hid_shape, action_dim):
        super().__init__()
        self.q1 = MLP_QNet(state_dim, hid_shape, action_dim)
        self.q2 = MLP_QNet(state_dim, hid_shape, action_dim)

    def forward(self, state, action):
        return self.q1(state, action), self.q2(state, action)

    def Q1(self, state, action):
        return self.q1(state, action)
    
class Agent:
    def __init__(self,env):
        torch.manual_seed(0)
        self.config = Config()
        self.state_dim = env.observation_space.shape[0]
        self.actor_hidden_layers = Config.actor_hidden_layers
        self.critic_hidden_layers = Config.critic_hidden_layers
        self.action_dim = env.action_space.shape[0]
        self.actor_learning_rate = Config.actor_learning_rate
        self.critic_learning_rate = Config.critic_learning_rate
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        '''Build Actor and Critic'''
        self.actor = Actor(self.state_dim,self.actor_hidden_layers,self.action_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr = self.actor_learning_rate)
        self.critic = Double_Q_Critic(self.state_dim,self.critic_hidden_layers,self.action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(params=self.critic.parameters(), lr = self.critic_learning_rate)
        self.model = [self.actor,self.critic,self.critic_target]
        self.optimizer = [self.actor_optimizer,self.critic_optimizer]
        self.algo = Algo(model = self.model, config = self.config,  optimizer = self.optimizer, device = self.device)

    def take_action(self,state):
        s = torch.tensor(state).view(1, self.state_dim).to(self.device)
        with torch.no_grad():
            action, logp_pi_a = self.actor.sample_act(s).cpu().numpy()[0]
        return action, logp_pi_a

    def update(self,transitions):
        actor_loss, critic_loss = self.algo.learn(transitions)
        return actor_loss, critic_loss
    
    def best_action(self,state):
        s = torch.tensor(state).view(1, self.state_dim).to(self.device)
        with torch.no_grad():
            action = self.actor.deterministic_act(s).cpu().numpy()[0]
        return action