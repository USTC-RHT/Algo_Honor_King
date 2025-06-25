import numpy as np
from prioritized_dqn_algo import Algo
from prioritized_dqn_config import Config
import sys

# 文件夹地址
upper_level_dir = 'E:\\Algo\\kaiwu_algo'
sys.path.append(upper_level_dir)
from common.segment_tree import SumSegmentTree,MinSegmentTree
from dataclasses import dataclass
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

# ReplayBuffer
class ReplayBuffer(object):
    def __init__(self, capacity):
        """Create Replay buffer.

        Parameters
        ----------
        capacity: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出
        self._maxsize = capacity
        self._next_idx = 0

    def add(self, state, action, reward, next_state, done):
        sample_data = SampleData(state, action, reward, next_state, done)
        self.buffer.append(sample_data)
        self._next_idx = (self._next_idx + 1) % self._maxsize
        return

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        """
        transitions = random.sample(self.buffer, batch_size)
        return transitions

    def size(self):
        return len(self.buffer)
    
# PrioritizedReplayBuffer
class PrioritizedReplayBuffer(ReplayBuffer):
    
    ''' 优先经验回放池 
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization or uniform case, 1 - full prioritization)
        ----------
    '''
    def __init__(self, capacity, alpha):
        super(PrioritizedReplayBuffer, self).__init__(capacity)
        self._alpha = alpha
        it_capacity = 1
        while it_capacity < capacity:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        idx = self._next_idx
        super().add(state, action, reward, next_state, done)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self.buffer) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        # 采集样本 i 的概率为 P(i)
        # P(i) = p_i ^ alpha / sum(p_j ^ alpha)
        p_min = self._it_min.min() / self._it_sum.sum()
        # importance-sampling (IS) weights
        # w_i = (N * P(i)) ^ (-beta)
        max_weight = (p_min * len(self.buffer)) ** (-beta)

        transitions = []
        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self.buffer)) ** (-beta)
            weights.append(weight / max_weight)
            transitions.append(self.buffer[idx])
        weights = np.array(weights)
        return transitions, weights, idxes

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self.buffer)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


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
        self.Q_target = Q_Net(self.state_dim,self.hidden_dim,self.action_dim).to(self.device)
        self.Q_target.load_state_dict(self.Q_main.state_dict())
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

    def update(self,transitions, weights):
        list_sample_data = transitions
        algo = Algo(model = self.model, config = Config, optimizer = self.optimizer, device = self.device)
        td_errors = algo.learn(list_sample_data, weights)
        return td_errors
    
    def best_action(self,state):
        state_tensor = torch.tensor(state).to(self.device)
        action = self.Q_main(state_tensor).argmax().item()
        return action
    

    
