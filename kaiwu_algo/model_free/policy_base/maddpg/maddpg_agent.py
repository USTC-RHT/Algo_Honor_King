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
    obs: np
    action: np
    reward: np
    next_obs: np
    dw: np

# ReplayBuffer
class ReplayBuffer():
    ''' 经验回放池 
        Max number of transitions to store in the buffer. 
        When the buffer overflows the old memories are dropped.
    '''
    def __init__(self, Config):
        self.agent_num = Config.agent_num
        self.buffer_size = Config.buffer_size
        self.batch_size = Config.batch_size
        self.count = 0
        self.current_size = 0
        self.buffer_obs_n, self.buffer_a_n, self.buffer_r_n, self.buffer_s_next_n, self.buffer_done_n = [], [], [], [], []
        for agent_id in range(self.agent_num):
            self.buffer_obs_n.append(np.empty((self.buffer_size, Config.obs_dim_n[agent_id])))
            self.buffer_a_n.append(np.empty((self.buffer_size, Config.action_dim_n[agent_id])))
            self.buffer_r_n.append(np.empty((self.buffer_size, 1)))
            self.buffer_s_next_n.append(np.empty((self.buffer_size, Config.obs_dim_n[agent_id])))
            self.buffer_done_n.append(np.empty((self.buffer_size, 1)))

    def store_transition(self, obs_n, a_n, r_n, obs_next_n, done_n):
        for agent_id in range(self.agent_num):
            self.buffer_obs_n[agent_id][self.count] = obs_n[agent_id]
            self.buffer_a_n[agent_id][self.count] = a_n[agent_id]
            self.buffer_r_n[agent_id][self.count] = r_n[agent_id]
            self.buffer_s_next_n[agent_id][self.count] = obs_next_n[agent_id]
            self.buffer_done_n[agent_id][self.count] = done_n[agent_id]
        self.count = (self.count + 1) % self.buffer_size
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def sample(self):
        index = np.random.choice(self.current_size, size=self.batch_size, replace=False) # 不允许重复抽样
        batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = [], [], [], [], []
        for agent_id in range(self.agent_num):
            batch_obs_n.append(torch.tensor(self.buffer_obs_n[agent_id][index], dtype=torch.float))
            batch_a_n.append(torch.tensor(self.buffer_a_n[agent_id][index], dtype=torch.float))
            batch_r_n.append(torch.tensor(self.buffer_r_n[agent_id][index], dtype=torch.float))
            batch_obs_next_n.append(torch.tensor(self.buffer_s_next_n[agent_id][index], dtype=torch.float))
            batch_done_n.append(torch.tensor(self.buffer_done_n[agent_id][index], dtype=torch.float))

        return batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n

# 正交初始化
def orthogonal_init(layer, gain=1.0):
    '''增益(gain) 控制正交矩阵的幅度'''
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)

# 策略网络构造
class Actor(torch.nn.Module):
    def __init__(self, obs_dim, hid_shape, action_dim, max_action, use_orthogonal_init = True):
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
        self.max_action = max_action

    def forward(self, x):
        ''' 映射到自定义的动作空间范围内 '''
        # return 0.5 * (self.max_action - self.min_action) * (self.Net(x) + 1) + self.min_action
        return self.max_action * self.Net(x)
    
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
    def __init__(self,agent_id):
        self.obs_dim_total = sum(Config.obs_dim_n)
        self.action_dim_total = sum(Config.action_dim_n)
        self.agent_id = agent_id
        '''MPE 环境的调用方式'''
        self.obs_dim = Config.obs_dim_n[agent_id]
        self.action_dim = Config.action_dim_n[agent_id]
        self.actor_hidden_layers = Config.actor_hidden_layers
        self.critic_hidden_layers = Config.critic_hidden_layers
        self.use_orthogonal_init = Config.use_orthogonal_init
        # self.min_action = Config.min_action
        self.max_action = Config.max_action
        # self.action_range = [self.min_action, self.max_action]
        self.noise = Config.noise
        self.actor_learning_rate = Config.actor_learning_rate
        self.critic_learning_rate = Config.critic_learning_rate
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        '''Build Actor and Critic'''
        self.actor = Actor(self.obs_dim,self.actor_hidden_layers,self.action_dim,self.max_action,self.use_orthogonal_init).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr = self.actor_learning_rate)
        self.critic = Critic(self.obs_dim_total,self.critic_hidden_layers,self.action_dim_total,self.use_orthogonal_init).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(params=self.critic.parameters(), lr = self.critic_learning_rate)
        self.model = [self.actor,self.actor_target,self.critic,self.critic_target]
        self.optimizer = [self.actor_optimizer,self.critic_optimizer]
        self.algo = Algo(model = self.model, config = Config,  optimizer = self.optimizer, device = self.device)

    def take_action(self,obs):
        s = torch.tensor(obs, dtype=torch.float32).view(1, self.obs_dim).to(self.device)
        with torch.no_grad():
            action = self.actor(s).cpu().numpy()[0]
            noise = np.random.normal(0, self.max_action * self.noise, size=self.action_dim)
        return (action + noise).clip(-self.max_action, self.max_action)

    def update(self,transitions,agent_n):
        actor_loss, critic_loss = self.algo.learn(transitions,agent_n,self.agent_id)
        return actor_loss, critic_loss
    
    def best_action(self,obs):
        s = torch.tensor(obs, dtype=torch.float32).view(1, self.obs_dim).to(self.device)
        with torch.no_grad():
            action = self.actor(s).cpu().numpy()[0]
        return action