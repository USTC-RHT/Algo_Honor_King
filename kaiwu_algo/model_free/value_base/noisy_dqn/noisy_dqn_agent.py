import numpy as np
from noisy_dqn_algo import Algo
from noisy_dqn_config import Config
from dataclasses import dataclass
import copy
import torch
import torch.nn as nn
import collections
import random
import sys
import os

cur_dir = os.path.dirname(__file__)
root = os.path.abspath(os.path.join(cur_dir, "kaiwu_algo"))
sys.path.append(root)
from common.rl_utils import NoisyLinear

# 数据类
@dataclass
class SampleData:
    state: int
    action: int
    reward: float
    next_state: int
    done: bool

# ReplayBuffer
class ReplayBuffer():
    
    ''' 经验回放池 
        Max number of transitions to store in the buffer. 
        When the buffer overflows the old memories are dropped.
    '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        sample_data = SampleData(state, action, reward, next_state, done)
        self.buffer.append(sample_data)
        return

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        return transitions

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)


# 构造神经网络
def build_net(layer_shape, activation, output_activation):
	'''Build networks with For loop'''
	layers = []
	for j in range(len(layer_shape)-1):
		if j < len(layer_shape) - 2: layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), activation()]
		else: layers += [NoisyLinear(layer_shape[j], layer_shape[j+1], sigma_init=0.25), output_activation()]
	return nn.Sequential(*layers)

class Noisy_Q_Net(nn.Module):
    def __init__(self, state_dim, hid_shape, action_dim):
        super(Noisy_Q_Net, self).__init__()
        layers = [state_dim] + list(hid_shape) + [action_dim]
        self.Q = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, s):
        q = self.Q(s)
        return q

    def transform_sample_data(self, list_sample_data, device):
        State = [sample_data.state for sample_data in list_sample_data]
        tensor_state = torch.tensor(np.stack(State)).to(device)  
        return tensor_state

# 构造智能体       
class Agent:
    def __init__(self):
        torch.manual_seed(0)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.state_dim = Config.state_dim
        self.hidden_layers = Config.hidden_layers
        self.action_dim = Config.action_dim
        self.learning_rate = Config.learning_rate
        self.Q_main = Noisy_Q_Net(self.state_dim,self.hidden_layers,self.action_dim).to(self.device)
        self.Q_target = copy.deepcopy(self.Q_main)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.Q_target.parameters(): p.requires_grad = False
        self.model = [self.Q_main,self.Q_target]
        self.epsilon = Config.epsilon
        self.optimizer = torch.optim.Adam(params=self.Q_main.parameters(), lr = self.learning_rate)

    def take_action(self, state):  # NoisyNet无需利用 epsilon-贪婪策略采取动作
        with torch.no_grad():
            state_tensor = torch.tensor(state).to(self.device)
            action = self.Q_main(state_tensor).argmax().item()
        return action

    def update(self,transitions):
        list_sample_data = transitions
        algo = Algo(model = self.model, config = Config, optimizer = self.optimizer, device = self.device)
        algo.learn(list_sample_data)
    
    def best_action(self,state):
        state_tensor = torch.tensor(state).to(self.device)
        action = self.Q_main(state_tensor).argmax().item()
        return action
    

    
