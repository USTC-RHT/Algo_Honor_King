import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ppo_continuous_algo import Algo
from ppo_continuous_config import Config
from dataclasses import dataclass
from torch.distributions import Normal, Beta

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
class GaussianActor(torch.nn.Module):
    def __init__(self, state_dim, hid_shape, action_dim):
        super(GaussianActor, self).__init__()
        # 1. 先构造共享层（隐藏层）
        layers = []
        layer_shape = [state_dim] + list(hid_shape)
        activation = nn.Tanh
        
        for j in range(len(layer_shape)-1):
            layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), activation()]
        
        # 用Sequential包装共享层
        self.shared_net = nn.Sequential(*layers)
        
        # 2. 分头：mu和sigma分别是两个线性层
        self.mu_head = nn.Linear(layer_shape[-1], action_dim)
        self.sigma_head = nn.Linear(layer_shape[-1], action_dim)

    def forward(self, x):
        x = self.shared_net(x)
        mu = torch.sigmoid(self.mu_head(x))            # 归一化到 0~1
        sigma = F.softplus(self.sigma_head(x))         # 保证标准差为正
        return mu, sigma
    
    def dist(self,x):
        mu, sigma = self.forward(x)
        return Normal(mu,sigma)
    
    def deterministic_act(self, state):
        mu, _ = self.forward(state)
        return mu
    
    def transform_sample_data(self, list_sample_data, device):
        State = []
        for sample_data in list_sample_data:
            State.append(sample_data.state)
            
        tensor_state = torch.tensor(np.array(State)).to(device)
        return tensor_state

class BetaActor(torch.nn.Module):
    def __init__(self, state_dim, hid_shape, action_dim):
        super(BetaActor, self).__init__()
        # 1. 先构造共享层（隐藏层）
        layers = []
        layer_shape = [state_dim] + list(hid_shape)
        activation = nn.Tanh
        
        for j in range(len(layer_shape)-1):
            layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), activation()]
        
        # 用Sequential包装共享层
        self.shared_net = nn.Sequential(*layers)
        
        # 2. 分头：mu和sigma分别是两个线性层
        self.alpha_head = nn.Linear(layer_shape[-1], action_dim)
        self.beta_head = nn.Linear(layer_shape[-1], action_dim)

    def forward(self, x):
        x = self.shared_net(x)
        alpha = F.softplus(self.alpha_head(x)) + 1.0
        beta = F.softplus(self.beta_head(x)) + 1.0          
        return alpha, beta
    
    def dist(self,x):
        alpha, beta = self.forward(x)
        return Beta(alpha, beta)
    
    def deterministic_act(self, x):
        alpha, beta = self.forward(x)
        mean = (alpha) / (alpha + beta)
        return mean
    
    def transform_sample_data(self, list_sample_data, device):
        State = []
        for sample_data in list_sample_data:
            State.append(sample_data.state)
            
        tensor_state = torch.tensor(np.array(State)).to(device)
        return tensor_state

# 价值网络构造
class Critic(torch.nn.Module):
    def __init__(self, state_dim, hid_shape):
        super(Critic, self).__init__()
        layers = []
        layer_shape = [state_dim] + list(hid_shape) + [1]
        '''设置激活函数为 Tanh '''
        activation = nn.Tanh
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
        self.action_dim = env.action_space.shape[0]
        self.actor_learning_rate = Config.actor_learning_rate
        self.critic_learning_rate = Config.critic_learning_rate
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        '''Build Actor and Critic'''
        self.actor = BetaActor(self.state_dim,self.actor_hidden_layers,self.action_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr = self.actor_learning_rate)
        self.critic = Critic(self.state_dim,self.critic_hidden_layers).to(self.device)
        self.critic_optimizer = torch.optim.Adam(params=self.critic.parameters(), lr = self.critic_learning_rate)
        self.model = [self.actor,self.critic]
        self.optimizer = [self.actor_optimizer,self.critic_optimizer]

    def take_action(self,state):
        # only used when interact with the env
        state = torch.tensor(state, dtype=torch.float32).view(1, self.state_dim).to(self.device)
        with torch.no_grad():
            dist = self.actor.dist(state)
            action = dist.sample()
            action = torch.clamp(action, 0, 1)
            logprob_a = dist.log_prob(action).cpu().numpy().flatten()
            return action.cpu().numpy()[0], logprob_a

    def update(self,Long_Traj):
        algo = Algo(model = self.model, config = Config,  optimizer = self.optimizer, device = self.device)
        actor_loss, critic_loss = algo.learn(Long_Traj)
        return actor_loss, critic_loss
    
    def best_action(self,state): 
        # only used when evaluate the policy.Making the performance more stable
        state = torch.tensor(state, dtype=torch.float32).view(1, self.state_dim).to(self.device)
        with torch.no_grad():
            mean = self.actor.deterministic_act(state)
            return mean.cpu().numpy()[0]

    
