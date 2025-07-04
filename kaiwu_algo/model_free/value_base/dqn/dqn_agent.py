import numpy as np
from dqn_algo import Algo
from dqn_config import Config
from dataclasses import dataclass
import torch
import torch.nn.functional as F
import collections
import random
import copy

'''Python 的 @dataclass 类型标注只是提示,不会强制类型检查或自动转换。'''
@dataclass
class SampleData:
    state: float
    action: int
    reward: float
    next_state: float
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


class Q_Net(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Q_Net, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)
    
    def transform_sample_data(self, list_sample_data, device):
        State = [sample_data.state for sample_data in list_sample_data]
        tensor_state = torch.tensor(np.stack(State)).to(device)  
        return tensor_state
        
class Agent:
    def __init__(self,env):
        torch.manual_seed(0)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.state_dim = env.observation_space.shape[0]
        self.hidden_dim = 128
        self.action_dim = env.action_space.n
        self.learning_rate = Config.learning_rate
        self.Q_main = Q_Net(self.state_dim,self.hidden_dim,self.action_dim).to(self.device)
        self.Q_target = copy.deepcopy(self.Q_main)
        self.model = [self.Q_main,self.Q_target]
        self.epsilon = Config.epsilon
        self.optimizer = torch.optim.Adam(params=self.Q_main.parameters(), lr = self.learning_rate)

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
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
    

    
