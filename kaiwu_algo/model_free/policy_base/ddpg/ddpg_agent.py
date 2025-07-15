import numpy as np
import collections
import random
import copy
import torch
import torch.nn as nn
from ddpg_algo import Algo
from ddpg_config import Config
from dataclasses import dataclass

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

# 策略网络构造
class Actor(torch.nn.Module):
    def __init__(self, state_dim, hid_shape, action_dim, max_action):
        super(Actor, self).__init__()
        layers = []
        layer_shape = [state_dim] + list(hid_shape) + [action_dim]
        '''设置激活函数为 ReLU '''
        activation = nn.ReLU
        '''Build networks with For loop'''
        for j in range(len(layer_shape)-1):
            if j < len(layer_shape) - 2: 
                layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), activation()]
            else: 
                layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), nn.Tanh()]
        self.mu = nn.Sequential(*layers)
        self.max_action = max_action        

    def forward(self, x):
        return self.mu(x) * self.max_action
    
    def transform_sample_data(self, list_sample_data, device):
        State = []
        for sample_data in list_sample_data:
            State.append(sample_data.state)
            
        tensor_state = torch.tensor(np.array(State)).to(device)
        return tensor_state

# 价值网络构造
class Critic(torch.nn.Module):
    def __init__(self, state_dim, hid_shape, action_dim):
        super(Critic, self).__init__()
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

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        return self.Q(x)
    
    def transform_sample_data(self, list_sample_data, device):
        State = []
        for sample_data in list_sample_data:
            State.append(sample_data.state)
            
        tensor_state = torch.tensor(np.array(State)).to(device)
        return tensor_state
    

class Agent:
    def __init__(self,env,max_action):
        torch.manual_seed(0)
        self.state_dim = env.observation_space.shape[0]
        self.actor_hidden_layers = Config.actor_hidden_layers
        self.critic_hidden_layers = Config.critic_hidden_layers
        self.action_dim = env.action_space.shape[0]
        self.max_action = max_action
        self.noise = Config.noise
        self.actor_learning_rate = Config.actor_learning_rate
        self.critic_learning_rate = Config.critic_learning_rate
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        '''Build Actor and Critic'''
        self.actor = Actor(self.state_dim,self.actor_hidden_layers,self.action_dim,self.max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr = self.actor_learning_rate)
        self.critic = Critic(self.state_dim,self.critic_hidden_layers,self.action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(params=self.critic.parameters(), lr = self.critic_learning_rate)
        self.model = [self.actor,self.actor_target,self.critic,self.critic_target]
        self.optimizer = [self.actor_optimizer,self.critic_optimizer]
        self.algo = Algo(model = self.model, config = Config,  optimizer = self.optimizer, device = self.device)

    def take_action(self,state):
        s = torch.tensor(state).view(1, self.state_dim).to(self.device)
        with torch.no_grad():
            action = self.actor(s).cpu().numpy()[0]
            noise = np.random.normal(0, self.max_action * self.noise, size=self.action_dim)
        return (action + noise).clip(-self.max_action, self.max_action)

    def update(self,transitions):
        actor_loss, critic_loss = self.algo.learn(transitions)
        return actor_loss, critic_loss
    
    def best_action(self,state):
        s = torch.tensor(state).view(1, self.state_dim).to(self.device)
        with torch.no_grad():
            action = self.actor(s).cpu().numpy()[0]
        return action