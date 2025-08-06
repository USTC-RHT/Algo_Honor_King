import numpy as np
import torch
import torch.nn as nn
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
        self.buffer = {'obs_n': np.zeros([self.batch_size, self.episode_limit, self.N, self.obs_dim]),
                       'state': np.zeros([self.batch_size, self.episode_limit, self.state_dim]),
                       'v_n': np.zeros([self.batch_size, self.episode_limit + 1, self.N]),
                       'avail_a_n': np.ones([self.batch_size, self.episode_limit, self.N, self.action_dim]),  # Note: We use 'np.ones' to initialize 'avail_a_n'
                       'a_n': np.zeros([self.batch_size, self.episode_limit, self.N]),
                       'logprob_a_n': np.zeros([self.batch_size, self.episode_limit, self.N]),
                       'r': np.zeros([self.batch_size, self.episode_limit, self.N]),
                       'dw': np.ones([self.batch_size, self.episode_limit, self.N]),  # Note: We use 'np.ones' to initialize 'dw'
                       'active': np.zeros([self.batch_size, self.episode_limit, self.N])
                       }
        self.episode_num = 0
        self.max_episode_len = 0

    def store_transition(self, episode_step, obs_n, state, v_n, avail_a_n, a_n, logprob_a_n, r, dw):
        self.buffer['obs_n'][self.episode_num][episode_step] = obs_n
        self.buffer['state'][self.episode_num][episode_step] = state
        self.buffer['v_n'][self.episode_num][episode_step] = v_n
        self.buffer['avail_a_n'][self.episode_num][episode_step] = avail_a_n
        self.buffer['a_n'][self.episode_num][episode_step] = a_n
        self.buffer['logprob_a_n'][self.episode_num][episode_step] = logprob_a_n
        self.buffer['r'][self.episode_num][episode_step] = np.array(r).repeat(self.N)
        self.buffer['dw'][self.episode_num][episode_step] = np.array(dw).repeat(self.N)

        self.buffer['active'][self.episode_num][episode_step] = np.ones(self.N)

    def store_last_value(self, episode_step, v_n):
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
class Q_network_RNN(nn.Module):
    def __init__(self, input_dim, hid_shape, action_dim, use_orthogonal_init = True):
        super(Q_network_RNN, self).__init__()
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

# VDN的全局价值网络构造   
class VDN_Net(nn.Module):
    def __init__(self, ):
        super(VDN_Net, self).__init__()

    def forward(self, q):
        return torch.sum(q, dim=-1, keepdim=True)  # (batch_size, max_episode_len, 1)
    

class Agent:
    def __init__(self):
        torch.manual_seed(0)
        self.state_dim = Config.state_dim
        self.obs_dim = Config.obs_dim_n[0]
        self.action_dim = Config.action_dim_n[0]
        self.agent_num = Config.agent_num
        self.hidden_layers = Config.hidden_layers
        self.learning_rate = Config.learning_rate
        self.use_agent_specific = Config.use_agent_specific
        self.use_lr_decay = Config.use_lr_decay
        self.add_last_action = Config.add_last_action
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
        self.actor = Actor(self.actor_input_dim,self.actor_hidden_layers,self.action_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr = self.actor_learning_rate)
        self.model = [self.actor,self.critic]
        self.optimizer = [self.actor_optimizer,self.critic_optimizer]
        self.algo = Algo(model = self.model, config = Config, optimizer = self.optimizer, device = self.device)

    def take_action(self,obs,avail_a):
        with torch.no_grad():
            obs = torch.tensor(obs).to(self.device)
            avail_a = torch.tensor(avail_a).to(self.device)
            probs = self.actor(obs, avail_a)
            action_dist = Categorical(probs=probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return [action.cpu().numpy(), log_prob.cpu().numpy()]
    
    def get_value(self,state,obs):
        ''' obs.shape=(N,obs_dim) '''
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        N = obs.shape[0]  # 获取 agent的数量 N
        # 扩展 state 到 [B, T, 1, state_dim] 再 repeat 到 [B, T, N, state_dim]
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        state = state.unsqueeze(0).repeat(N,1)
        with torch.no_grad():    
            if self.use_agent_specific:  # Add local obs of the agent
                critic_input = torch.cat([state, obs], dim=-1)
            else:
                critic_input = state
            value = self.critic(critic_input)
            return value.cpu().numpy().squeeze()

    def update(self,batch,total_steps):
        actor_loss, critic_loss = self.algo.learn(batch)
        if self.use_lr_decay:
            self.lr_decay(total_steps)
        return actor_loss, critic_loss
    
    def lr_decay(self, total_steps):
        self.actor_learning_rate *= 1 - total_steps / self.Max_train_steps
        self.critic_learning_rate *= 1 - total_steps / self.Max_train_steps
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = self.actor_learning_rate    
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = self.critic_learning_rate
        return   
    
    def best_action(self,obs,avail_a):
        with torch.no_grad():
            obs = torch.tensor(obs).to(self.device)
            avail_a = torch.tensor(avail_a).to(self.device)
            probs = self.actor(obs, avail_a).cpu().numpy()
            action = np.argmax(probs, axis=-1)
        return action