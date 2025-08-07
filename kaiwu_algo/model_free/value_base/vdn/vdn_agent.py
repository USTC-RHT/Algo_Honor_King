import copy
import torch
import torch.nn as nn
import numpy as np
from vdn_algo import Algo
from vdn_config import Config
from torch.distributions import Categorical
from common.rl_utils import orthogonal_init

class ReplayBuffer:
    def __init__(self):
        self.N = Config.agent_num
        self.obs_dim = Config.obs_dim_n[0]
        self.state_dim = Config.state_dim
        self.action_dim = Config.action_dim_n[0]
        self.episode_limit = Config.episode_limit
        self.batch_size = Config.batch_size
        self.episode_num = 0
        self.max_episode_len = 0
        self.buffer = None
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer = {'obs_n': np.zeros([self.batch_size, self.episode_limit + 1, self.N, self.obs_dim]),
                       'avail_a_n': np.ones([self.batch_size, self.episode_limit, self.N, self.action_dim]),  # Note: We use 'np.ones' to initialize 'avail_a_n'
                       'a_n': np.zeros([self.batch_size, self.episode_limit, self.N]),
                       'last_onehot_a_n': np.zeros([self.batch_size, self.episode_limit + 1, self.N, self.action_dim]),
                       'r': np.zeros([self.batch_size, self.episode_limit, self.N]),
                       'dw': np.ones([self.batch_size, self.episode_limit, self.N]),  # Note: We use 'np.ones' to initialize 'dw'
                       'active': np.zeros([self.batch_size, self.episode_limit, self.N])
                       }
        self.episode_num = 0
        self.max_episode_len = 0

    def store_transition(self, episode_step, obs_n, avail_a_n, a_n, last_onehot_a_n, r, dw):
        self.buffer['obs_n'][self.episode_num][episode_step] = obs_n
        self.buffer['avail_a_n'][self.episode_num][episode_step] = avail_a_n
        self.buffer['a_n'][self.episode_num][episode_step] = a_n
        self.buffer['last_onehot_a_n'][self.episode_num][episode_step + 1] = last_onehot_a_n
        self.buffer['r'][self.episode_num][episode_step] = np.array(r).repeat(self.N)
        self.buffer['dw'][self.episode_num][episode_step] = np.array(dw).repeat(self.N)

        self.buffer['active'][self.episode_num][episode_step] = np.ones(self.N)

    def store_last_value(self, episode_step, obs_n, avail_a_n):
        self.buffer['v_n'][self.episode_num][episode_step] = v_n
        self.episode_num += 1
        # Record max_episode_len
        if episode_step > self.max_episode_len:
            self.max_episode_len = episode_step

    def get_training_data(self):
        batch = {}
        for key in self.buffer.keys():
            if key == 'a_n':
                batch[key] = torch.tensor(self.buffer[key][:, :self.max_episode_len], dtype=torch.long)
            elif key == 'v_n':
                batch[key] = torch.tensor(self.buffer[key][:, :self.max_episode_len + 1], dtype=torch.float32)
            else:
                batch[key] = torch.tensor(self.buffer[key][:, :self.max_episode_len], dtype=torch.float32)
        batch['max_episode_len'] = self.max_episode_len
        return batch
    


# 每个智能体的价值网络构造
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

class Agent:
    def __init__(self):
        torch.manual_seed(0)
        self.epsilon = Config.epsilon
        self.obs_dim = Config.obs_dim_n[0]
        self.action_dim = Config.action_dim_n[0]
        self.agent_num = Config.agent_num
        self.hidden_layers = Config.hidden_layers
        self.learning_rate = Config.learning_rate
        self.use_agent_specific = Config.use_agent_specific
        self.use_lr_decay = Config.use_lr_decay
        self.add_last_action = Config.add_last_action
        self.add_agent_id = Config.add_agent_id
        self.Max_train_steps = Config.Max_train_steps
        
        # Compute the input dimension
        self.input_dim = self.obs_dim
        if self.add_last_action:
            print("------add last action------")
            self.input_dim += self.action_dim
        if self.add_agent_id:
            print("------add agent id------")
            self.input_dim += self.agent_num

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.Q_main = Q_Net(self.input_dim,self.hidden_layers,self.action_dim).to(self.device)
        self.Q_target = copy.deepcopy(self.Q_main)
        self.model = [self.Q_main,self.Q_target]
        self.optimizer = torch.optim.Adam(params=self.Q_main.parameters(), lr = self.learning_rate)
        self.algo = Algo(model = self.model, config = Config, optimizer = self.optimizer, device = self.device)

    def take_action(self,obs_n,avail_a_n,last_onehot_a_n):
        # obs.shape=(N,obs_dim)
        if np.random.uniform() < self.epsilon:  # epsilon-greedy
            # Only available actions can be chosen
            a_n = [np.random.choice(np.nonzero(avail_a)[0]) for avail_a in avail_a_n]
        else:
            a_n = self.best_action(obs_n,avail_a_n,last_onehot_a_n)
        return a_n

    def update(self,batch,total_steps):
        actor_loss, critic_loss = self.algo.learn(batch)
        if self.use_lr_decay:
            self.lr_decay(total_steps)
        return actor_loss, critic_loss
    
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