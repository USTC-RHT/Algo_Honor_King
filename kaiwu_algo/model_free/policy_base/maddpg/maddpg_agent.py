import numpy as np
import collections
import random
import copy
import torch
import torch.nn as nn
from maddpg_algo import Algo
from maddpg_config import Config
from dataclasses import dataclass

@dataclass
class SampleData:
    obs: dict
    action: dict
    reward: dict
    next_obs: dict
    dw: dict

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

# 正交初始化
def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)

# 策略网络构造
class Actor(torch.nn.Module):
    def __init__(self, obs_dim, hid_shape, action_dim, action_range, use_orthogonal_init = True):
        super(Actor, self).__init__()
        layers = []
        layer_shape = [obs_dim] + list(hid_shape) + [action_dim]
        ''' 设置激活函数为 ReLU '''
        activation = nn.ReLU
        ''' Build networks with For loop '''
        for j in range(len(layer_shape)-1):
            if j < len(layer_shape) - 2: 
                layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), activation()]
            else: 
                layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), nn.Tanh()]
        self.Net = nn.Sequential(*layers)
        ''' 设置正交初始化 '''
        if use_orthogonal_init:
            for layer in self.Net:
                if isinstance(layer, nn.Linear):
                    orthogonal_init(layer)
        self.min_action, self.max_action = action_range 

    def forward(self, x):
        ''' 映射到自定义的动作空间范围内 '''
        return 0.5 * (self.max_action - self.min_action) * (self.Net(x) + 1) + self.min_action
    
    def transform_sample_data(self, list_sample_data, device):
        State = []
        for sample_data in list_sample_data:
            State.append(sample_data.state)
            
        tensor_state = torch.tensor(np.array(State)).to(device)
        return tensor_state

# 价值网络构造
class Critic(torch.nn.Module):
    def __init__(self, obs_dim_total, hid_shape, action_dim_total, use_orthogonal_init = True):
        super(Critic, self).__init__()
        layers = []
        '''中心化的动作价值函数:所有智能体要同时给出自己的观测和相应的动作'''
        layer_shape = [obs_dim_total + action_dim_total] + list(hid_shape) + [1]
        '''设置激活函数为 ReLU '''
        activation = nn.ReLU
        '''Build networks with For loop'''
        for j in range(len(layer_shape)-1):
            if j < len(layer_shape) - 2: 
                layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), activation()]
            else: 
                layers += [nn.Linear(layer_shape[j], layer_shape[j+1])]
        self.Q = nn.Sequential(*layers)    
        ''' 设置正交初始化 '''
        if use_orthogonal_init:
            for layer in self.Q:
                if isinstance(layer, nn.Linear):
                    orthogonal_init(layer)       

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        return self.Q(x)
    
    def transform_sample_data(self, list_sample_data, device):
        State = []
        for sample_data in list_sample_data:
            State.append(sample_data.state)
            
        tensor_state = torch.tensor(np.array(State)).to(device)
        return tensor_state
    
'''单个智能体对应的类'''
class Agent:
    def __init__(self,env,agent_name):
        torch.manual_seed(0)
        self.obs_dim_total = 0
        self.action_dim_total = 0
        self.agent_names = Config.agent_names
        for name in Config.agent_names:
            self.obs_dim_total += env.observation_space(name).shape[0]
            self.action_dim_total += env.action_space(name).shape[0]
        self.agent_name = agent_name
        '''MPE 环境的调用方式'''
        self.obs_dim = env.observation_space(agent_name).shape[0]
        self.action_dim = env.action_space(agent_name).shape[0]
        self.actor_hidden_layers = Config.actor_hidden_layers
        self.critic_hidden_layers = Config.critic_hidden_layers
        self.use_orthogonal_init = Config.use_orthogonal_init
        self.min_action = Config.min_action
        self.max_action = Config.max_action
        self.action_range = [self.min_action, self.max_action]
        self.noise = Config.noise
        self.actor_learning_rate = Config.actor_learning_rate
        self.critic_learning_rate = Config.critic_learning_rate
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        '''Build Actor and Critic'''
        self.actor = Actor(self.obs_dim,self.actor_hidden_layers,self.action_dim,self.action_range,self.use_orthogonal_init).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr = self.actor_learning_rate)
        self.critic = Critic(self.obs_dim_total,self.critic_hidden_layers,self.action_dim_total,self.use_orthogonal_init).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(params=self.critic.parameters(), lr = self.critic_learning_rate)
        self.model = [self.actor,self.actor_target,self.critic,self.critic_target]
        self.optimizer = [self.actor_optimizer,self.critic_optimizer]
        self.algo = Algo(model = self.model, config = Config,  optimizer = self.optimizer, device = self.device)

    def take_action(self,obs):
        s = torch.tensor(obs).view(1, self.obs_dim).to(self.device)
        with torch.no_grad():
            action = self.actor(s).cpu().numpy()[0]
            
            noise = np.random.normal(0, self.max_action * self.noise, size=self.action_dim)
        return (action + noise).clip(self.min_action, self.max_action)

    def update(self,transitions,agent_n):
        actor_loss, critic_loss = self.algo.learn(transitions,agent_n,self.agent_names,self.agent_name)
        return actor_loss, critic_loss
    
    def best_action(self,obs):
        s = torch.tensor(obs).view(1, self.obs_dim).to(self.device)
        with torch.no_grad():
            action = self.actor(s).cpu().numpy()[0]
        return action