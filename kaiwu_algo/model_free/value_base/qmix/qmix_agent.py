import copy
import torch
import torch.nn as nn
import numpy as np
from qmix_algo import Algo
from qmix_config import Config
from common.rl_utils import orthogonal_init
import torch.nn.functional as F

class ReplayBuffer:
    def __init__(self):
        self.N = Config.agent_num
        self.obs_dim = Config.obs_dim_n[0]
        self.state_dim = Config.state_dim
        self.action_dim = Config.action_dim_n[0]
        self.episode_limit = Config.episode_limit
        self.buffer_size = Config.buffer_size
        self.episode_num = 0
        self.current_size = 0
        self.max_episode_len = 0
        self.buffer = None
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer = {'obs_n': np.zeros([self.buffer_size, self.episode_limit + 1, self.N, self.obs_dim]),
                       'state': np.zeros([self.buffer_size, self.episode_limit + 1, self.state_dim]),
                       'avail_a_n': np.ones([self.buffer_size, self.episode_limit + 1, self.N, self.action_dim]),
                       'a_n': np.zeros([self.buffer_size, self.episode_limit, self.N]),
                       'last_onehot_a_n': np.zeros([self.buffer_size, self.episode_limit + 1, self.N, self.action_dim]),
                       'r': np.zeros([self.buffer_size, self.episode_limit, 1]),
                       'dw': np.ones([self.buffer_size, self.episode_limit, 1]),
                       'active': np.zeros([self.buffer_size, self.episode_limit, 1])
                       }
        self.episode_len = np.zeros(self.buffer_size)
        self.episode_num = 0

    def store_transition(self, episode_step, obs_n, state, avail_a_n, a_n, last_onehot_a_n, r, dw):
        self.buffer['obs_n'][self.episode_num][episode_step] = obs_n
        self.buffer['state'][self.episode_num][episode_step] = state
        self.buffer['avail_a_n'][self.episode_num][episode_step] = avail_a_n
        self.buffer['a_n'][self.episode_num][episode_step] = a_n
        self.buffer['last_onehot_a_n'][self.episode_num][episode_step + 1] = last_onehot_a_n
        self.buffer['r'][self.episode_num][episode_step] = r
        self.buffer['dw'][self.episode_num][episode_step] = dw

        self.buffer['active'][self.episode_num][episode_step] = 1.0

    def store_last_value(self, episode_step, obs_n, state, avail_a_n):
        self.buffer['obs_n'][self.episode_num][episode_step] = obs_n
        self.buffer['state'][self.episode_num][episode_step] = state
        self.buffer['avail_a_n'][self.episode_num][episode_step] = avail_a_n
        self.episode_len[self.episode_num] = episode_step
        ''' 队列,先进先出 '''
        self.episode_num = (self.episode_num + 1) % self.buffer_size
        self.current_size = min(self.current_size + 1, self.buffer_size)
        if episode_step > self.max_episode_len:
            self.max_episode_len = episode_step

    def sample(self, batch_size):  # 从buffer中采样数据,数量为 batch_size
        index = np.random.choice(self.current_size, size=batch_size, replace=False)
        max_episode_len = self.max_episode_len
        batch = {}
        for key in self.buffer.keys():
            if key in ['obs_n','state','avail_a_n','last_onehot_a_n']:
                batch[key] = torch.tensor(self.buffer[key][index, :max_episode_len + 1], dtype=torch.float32)
            elif key == 'a_n':
                batch[key] = torch.tensor(self.buffer[key][index, :max_episode_len], dtype=torch.long)
            else:
                batch[key] = torch.tensor(self.buffer[key][index, :max_episode_len], dtype=torch.float32)
        batch['max_episode_len'] = max_episode_len
        return batch
    
# 智能体的价值网络构造
class Q_Net(nn.Module):
    def __init__(self, input_dim, hid_shape, action_dim, use_orthogonal_init = True):
        super(Q_Net, self).__init__()
        self.rnn_hidden = None
        '''设置激活函数为 ReLU '''
        self.activate_func = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, hid_shape[0])
        self.rnn = nn.GRUCell(hid_shape[0], hid_shape[1])
        self.fc2 = nn.Linear(hid_shape[1], action_dim)
        if use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2)

    def forward(self, x):
        ''' When 'choose_action', inputs.shape(N, input_dim)
        When 'train', inputs.shape(bach_size * N,input_dim) '''
        x1 = self.activate_func(self.fc1(x))
        self.rnn_hidden = self.rnn(x1, self.rnn_hidden)
        Q = self.fc2(self.rnn_hidden)
        return Q


class QMIX_Net(nn.Module):
    def __init__(self):
        super(QMIX_Net, self).__init__()
        self.N = Config.agent_num
        self.state_dim = Config.state_dim
        self.batch_size = Config.batch_size
        self.qmix_hidden_dim = Config.qmix_hidden_dim
        self.hyper_hidden_dim = Config.hyper_hidden_dim
        """
        在 QMIX 中,Mixing Network 的参数（包括权重和偏置）
        都是由全局状态 state 通过超网络(hypernetwork)动态生成的。

        w1:state_dim -> N * qmix_hidden_dim -> 作为Mix网络的第一层权重矩阵 (N,qmix_hidden_dim)
        b1:state_dim -> 1 * qmix_hidden_dim -> 作为Mix网络的第一层偏置向量 (1,qmix_hidden_dim)
        w2:state_dim -> 1 * qmix_hidden_dim -> 作为Mix网络的第二层权重矩阵 (qmix_hidden_dim,1)
        b2:state_dim -> 1 * 1 -> 作为Mix网络的第二层偏置向量 (1,1)
        """
        self.hyper_w1 = nn.Linear(self.state_dim, self.N * self.qmix_hidden_dim)
        self.hyper_w2 = nn.Linear(self.state_dim, self.qmix_hidden_dim * 1)

        self.hyper_b1 = nn.Linear(self.state_dim, self.qmix_hidden_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(self.state_dim, self.qmix_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.qmix_hidden_dim, 1))

    def forward(self, q, s):
        # q.shape(batch_size, max_episode_len, N)
        # s.shape(batch_size, max_episode_len,state_dim)
        q = q.view(-1, 1, self.N)  # (batch_size * max_episode_len, 1, N)
        s = s.reshape(-1, self.state_dim)  # (batch_size * max_episode_len, state_dim)

        w1 = torch.abs(self.hyper_w1(s))  # (batch_size * max_episode_len, N * qmix_hidden_dim)
        b1 = self.hyper_b1(s)  # (batch_size * max_episode_len, qmix_hidden_dim)
        w1 = w1.view(-1, self.N, self.qmix_hidden_dim)  # (batch_size * max_episode_len, N,  qmix_hidden_dim)
        b1 = b1.view(-1, 1, self.qmix_hidden_dim)  # (batch_size * max_episode_len, 1, qmix_hidden_dim)

        # torch.bmm 批量矩阵乘法（batch matrix multiplication）函数,会对 batch 里的每一对矩阵分别做矩阵乘法
        q_hidden = F.elu(torch.bmm(q, w1) + b1)  # (batch_size * max_episode_len, 1, qmix_hidden_dim)

        w2 = torch.abs(self.hyper_w2(s))  # (batch_size * max_episode_len, qmix_hidden_dim * 1)
        b2 = self.hyper_b2(s)  # (batch_size * max_episode_len,1)
        w2 = w2.view(-1, self.qmix_hidden_dim, 1)  # (batch_size * max_episode_len, qmix_hidden_dim, 1)
        b2 = b2.view(-1, 1, 1)  # (batch_size * max_episode_len, 1， 1)

        q_total = torch.bmm(q_hidden, w2) + b2  # (batch_size * max_episode_len, 1， 1)
        q_total = q_total.view(self.batch_size, -1, 1)  # (batch_size, max_episode_len, 1)
        return q_total


class Agent:
    ''' 采用参数共享方式,用同一个动作价值网络
    泛化能力强: 共享网络能学到"通用策略",适应不同 agent,尤其适合同质agent。
    数据利用率高: 所有 agent 的经验都能用来更新同一个网络，收敛更快。
    参数量小,易于扩展: 只需存储和更新一个网络，节省内存和计算资源。'''
    def __init__(self):
        torch.manual_seed(0)
        self.epsilon = Config.epsilon
        self.obs_dim = Config.obs_dim_n[0]
        self.action_dim = Config.action_dim_n[0]
        self.agent_num = Config.agent_num
        self.hidden_layers = Config.hidden_layers
        self.learning_rate = Config.learning_rate
        self.use_lr_decay = Config.use_lr_decay
        self.add_last_action = Config.add_last_action
        self.add_agent_id = Config.add_agent_id
        self.Max_train_steps = Config.Max_train_steps
        
        # Compute the input dimension
        self.input_dim = self.obs_dim
        if self.add_last_action:
            '''让动作价值网络能够利用"动作-状态历史",提升表达能力,尤其是在部分可观测环境下。'''
            print("------add last action------")
            self.input_dim += self.action_dim
        if self.add_agent_id:
            '''为了让共享参数的动作价值网络能够区分不同的 agent,从而学出个性化的策略。'''
            print("------add agent id------")
            self.input_dim += self.agent_num

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.Q_main = Q_Net(self.input_dim,self.hidden_layers,self.action_dim).to(self.device)
        self.Q_target = copy.deepcopy(self.Q_main)
        self.QMIX_main = QMIX_Net().to(self.device)
        self.QMIX_target = copy.deepcopy(self.QMIX_main)
        self.model = [self.Q_main,self.Q_target,self.QMIX_main,self.QMIX_target]
        self.parameters = list(self.QMIX_main.parameters()) + list(self.Q_main.parameters())
        self.optimizer = torch.optim.Adam(params=self.parameters, lr = self.learning_rate)
        self.algo = Algo(model = self.model, config = Config, optimizer = self.optimizer, device = self.device)

    def take_action(self,obs_n,avail_a_n,last_onehot_a_n):
        # obs.shape=(N,obs_dim)
        if np.random.uniform() < self.epsilon:  # epsilon-greedy
            # Only available actions can be chosen
            a_n = [np.random.choice(np.nonzero(avail_a)[0]) for avail_a in avail_a_n]
        else:
            a_n = self.best_action(obs_n,avail_a_n,last_onehot_a_n)
        # epsilon decay
        self.epsilon = max(self.epsilon - Config.epsilon_decay, Config.epsilon_min)
        return a_n

    def update(self,batch,total_steps):
        loss = self.algo.learn(batch)
        if self.use_lr_decay:
            self.lr_decay(total_steps)
        return loss
    
    def lr_decay(self, total_steps):
        self.learning_rate *= 1 - total_steps / self.Max_train_steps
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate
        return   
    
    def best_action(self,obs_n,avail_a_n,last_onehot_a_n):
        with torch.no_grad():
            inputs = [torch.tensor(obs_n, dtype=torch.float32)]
            if self.add_last_action:
                inputs.append(torch.tensor(last_onehot_a_n, dtype=torch.float32))
            if self.add_agent_id:
                inputs.append(torch.eye(self.agent_num))

            inputs = torch.cat([x for x in inputs], dim=-1)  # inputs.shape=(N,inputs_dim)
            q_value = self.Q_main(inputs)

            avail_a_n = torch.tensor(avail_a_n, dtype=torch.float32)  # avail_a_n.shape=(N, action_dim)
            q_value[avail_a_n == 0] = -float('inf')  # Mask the unavailable actions
            a_n = q_value.argmax(dim = -1).numpy()
        return a_n