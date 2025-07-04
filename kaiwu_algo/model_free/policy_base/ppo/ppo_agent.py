import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ppo_algo import Algo
from ppo_config import Config
from dataclasses import dataclass

'''s, a, r, s_next, logprob_a, dw, done'''
@dataclass
class SampleData:
    state: float
    action: int
    reward: float
    next_state: float
    logprob_a: float
    dw: bool
    done: bool

# 策略网络构造
class Actor(torch.nn.Module):
    def __init__(self, state_dim, hid_shape, action_dim):
        super(Actor, self).__init__()
        layers = []
        layer_shape = [state_dim] + list(hid_shape) + [action_dim]
        '''设置激活函数为 Tanh '''
        activation = nn.Tanh
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

# 价值网络构造
class Critic(torch.nn.Module):
    def __init__(self, state_dim, hid_shape):
        super(Actor, self).__init__()
        layers = []
        layer_shape = [state_dim] + list(hid_shape) + [1]
        '''设置激活函数为 Tanh '''
        activation = nn.ReLU
        '''Build networks with For loop'''
        for j in range(len(layer_shape)-1):
            if j < len(layer_shape) - 2: 
                layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), activation()]
            else: 
                layers += [nn.Linear(layer_shape[j], layer_shape[j+1])]
        self.Q = nn.Sequential(*layers)        

    def forward(self, x):
        return self.Q(x)
    
    def transform_sample_data(self, list_sample_data, device):
        State = []
        for sample_data in list_sample_data:
            State.append(sample_data.state)
            
        tensor_state = torch.tensor(np.array(State)).to(device)
        return tensor_state
    

class Agent:
    def __init__(self,env):
        torch.manual_seed(0)
        self.state_dim = env.observation_space.shape[0]
        self.actor_hidden_layers = Config.actor_hidden_layers
        self.critic_hidden_layers = Config.critic_hidden_layers
        self.action_dim = env.action_space.n
        self.actor_learning_rate = Config.actor_learning_rate
        self.critic_learning_rate = Config.critic_learning_rate
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        '''Build Actor and Critic'''
        self.actor = Actor(self.state_dim,self.actor_hidden_layers,self.action_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(params=self.model.parameters(), lr = self.actor_learning_rate)
        self.critic = Critic(self.state_dim,self.critic_hidden_layers,self.action_dim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(params=self.model.parameters(), lr = self.critic_learning_rate)
        self.model = [self.actor,self.critic]
        self.optimizer = [self.actor_optimizer,self.critic_optimizer]

    def take_action(self,state):
        s = torch.tensor(state).view(1, self.state_dim).to(self.device)
        # 推理得到的结果已经是概率分布
        with torch.no_grad():
            probs = self.actor(s)
            action_dist = torch.distributions.Categorical(probs=probs)
        action = action_dist.sample()
        return action.item()

    def update(self,Episode):
        # 创建一个 Data 实例
        list_sample_data = [SampleData(state=obs, action=action, reward=r) for (obs,action,r) in Episode]
        algo = Algo(model = self.model, config = Config,  optimizer = self.optimizer, device = self.device)
        algo.learn(list_sample_data)
    
    def best_action(self,state):
        with torch.no_grad():
            s = torch.tensor(state).view(1, self.state_dim).to(self.device)
            action = np.argmax(self.model(s).detach().cpu().numpy())
        return action
    

    
