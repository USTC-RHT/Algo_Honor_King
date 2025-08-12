import numpy as np
from dueling_dqn_algo import Algo
from dueling_dqn_config import Config
from dataclasses import dataclass
import copy
import torch
import torch.nn.functional as F
import collections
import random

@dataclass
class SampleData:
    state: int
    action: int
    reward: float
    next_state: int
    done: bool

# ReplayBuffer, PrioritizedReplayBuffer
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

class VAnet(torch.nn.Module):
    ''' 只有一层隐藏层的A(Advantage)网络和V(Value)网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(VAnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  # 共享网络部分
        self.fc_A = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_V = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        A = self.fc_A(F.relu(self.fc1(x)))
        V = self.fc_V(F.relu(self.fc1(x)))
        ''' 需要减去Advantage的最大值或均值,否则公式具有不唯一性 Q = V + A
        优势函数只需跟随均值变化，不用频繁补偿最优动作的变化，让优化过程更加稳定 '''
        Q = V + A - A.mean()              # Q值由V值和A值计算得到
        return Q
    
    def transform_sample_data(self, list_sample_data, device):
        State = [sample_data.state for sample_data in list_sample_data]
        tensor_state = torch.tensor(np.stack(State)).to(device)  
        return tensor_state

class Agent:
    def __init__(self):
        torch.manual_seed(0)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.state_dim = Config.state_dim
        self.hidden_dim = Config.hidden_dim
        self.action_dim = Config.action_dim
        self.learning_rate = Config.learning_rate
        self.Q_main = VAnet(self.state_dim,self.hidden_dim,self.action_dim).to(self.device)
        self.Q_target = copy.deepcopy(self.Q_main)
        self.model = [self.Q_main,self.Q_target]
        self.epsilon = Config.epsilon
        self.optimizer = torch.optim.Adam(params=self.Q_main.parameters(), lr = self.learning_rate)
        self.algo = Algo(model = self.model, config = Config, optimizer = self.optimizer, device = self.device)

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state_tensor = torch.tensor(state).to(self.device)
            action = self.Q_main(state_tensor).argmax().item()
        return action

    def update(self,transitions):
        list_sample_data = transitions
        self.algo.learn(list_sample_data)
        return
    
    def best_action(self,state):
        state_tensor = torch.tensor(state).to(self.device)
        action = self.Q_main(state_tensor).argmax().item()
        return action
    
    def max_q_value(self,state):
        state_tensor = torch.tensor(state).to(self.device)
        q_max = max(self.Q_main(state_tensor))   
        return q_max.item()
    

    
