import numpy as np
import collections
import random
import copy
import torch
import torch.nn as nn
from sac_discrete_algo import Algo
from sac_discrete_config import Config
from dataclasses import dataclass
from torch.distributions import Categorical

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
        layer_shape = [state_dim] + list(hid_shape) + [action_dim]
        '''设置激活函数为 ReLU '''
        activation = nn.ReLU
        '''Build networks with For loop'''
        for j in range(len(layer_shape)-1):
            if j < len(layer_shape) - 2: 
                layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), activation()]
            else: 
                layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), nn.Softmax(dim=1)]
        self.Pi = nn.Sequential(*layers)   
     
    def forward(self, x):
        return self.Pi(x)
    
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
        layer_shape = [state_dim] + list(hid_shape) + [action_dim]
        '''设置激活函数为 ReLU '''
        activation = nn.ReLU
        '''Build networks with For loop'''
        for j in range(len(layer_shape)-1):
            if j < len(layer_shape) - 2: 
                layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), activation()]
            else: 
                layers += [nn.Linear(layer_shape[j], layer_shape[j+1])]
        self.Q = nn.Sequential(*layers)  

    def forward(self, state):
        return self.Q(state)

# Soft价值网络构造
class Double_Q_Critic(nn.Module):
    def __init__(self, state_dim, hid_shape, action_dim):
        super().__init__()
        self.q1 = MLP_QNet(state_dim, hid_shape, action_dim)
        self.q2 = MLP_QNet(state_dim, hid_shape, action_dim)

    def forward(self, state):
        return self.q1(state), self.q2(state)
    
class Agent:
    def __init__(self,env):
        torch.manual_seed(0)
        self.config = Config()
        self.alpha = Config.alpha
        self.adaptive_alpha = Config.adaptive_alpha
        self.state_dim = env.observation_space.shape[0]
        self.actor_hidden_layers = Config.actor_hidden_layers
        self.critic_hidden_layers = Config.critic_hidden_layers
        self.action_dim = env.action_space.n
        self.config.action_dim = self.action_dim
        self.actor_learning_rate = Config.actor_learning_rate
        self.critic_learning_rate = Config.critic_learning_rate
        self.log_alpha_learning_rate = Config.log_actor_learning_rate
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        '''Build Actor and Critic'''
        self.actor = Actor(self.state_dim,self.actor_hidden_layers,self.action_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr = self.actor_learning_rate)
        self.critic = Double_Q_Critic(self.state_dim,self.critic_hidden_layers,self.action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(params=self.critic.parameters(), lr = self.critic_learning_rate)

        if self.adaptive_alpha:
            # Use 0.6 because the recommended 0.98 will cause alpha explosion.
            ''' -np.log(1 / self.action_dim)是均匀分布的熵 '''
            self.config.target_entropy = 0.6 * (-np.log(1 / self.action_dim))  # H(discrete)>0
            # learn log_alpha instead of alpha to ensure : alpha > 0
            self.log_alpha = torch.nn.Parameter(torch.tensor(np.log(self.alpha), dtype=torch.float32, requires_grad=True, device=self.device))
            self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.log_alpha_learning_rate)

        self.model = [self.actor,self.critic,self.critic_target,self.log_alpha]
        self.optimizer = [self.actor_optimizer,self.critic_optimizer,self.log_alpha_optimizer]
        self.algo = Algo(model = self.model, config = self.config,  optimizer = self.optimizer, device = self.device)

    def take_action(self,state):
        with torch.no_grad():
            s = torch.tensor(state).view(1, self.state_dim).to(self.device)
            probs = self.actor(s)
            action_dist = Categorical(probs=probs)
        action = action_dist.sample()
        return action.item()

    def update(self,transitions):
        actor_loss, critic_loss = self.algo.learn(transitions)
        return actor_loss, critic_loss
    
    def best_action(self,state):
        with torch.no_grad():
            s = torch.tensor(state).view(1, self.state_dim).to(self.device)
            action = np.argmax(self.actor(s).detach().cpu().numpy())
        return action