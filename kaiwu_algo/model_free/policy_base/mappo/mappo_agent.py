import numpy as np
import torch
import torch.nn as nn
from mappo_algo import Algo
from mappo_config import Config
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
        return batch
    
# 策略网络构造
class Actor(torch.nn.Module):
    def __init__(self, actor_input_dim, hid_shape, action_dim, use_orthogonal_init = True):
        super(Actor, self).__init__()
        layers = []
        layer_shape = [actor_input_dim] + list(hid_shape) + [action_dim]
        '''设置激活函数为 ReLU '''
        activation = nn.ReLU
        '''Build networks with For loop'''
        for j in range(len(layer_shape)-1):
            if j < len(layer_shape) - 2: 
                layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), activation()]
            else: 
                layers += [nn.Linear(layer_shape[j], layer_shape[j+1])]
        self.model = nn.Sequential(*layers)   
        ''' 设置正交初始化 '''
        if use_orthogonal_init:
            for layer in self.model:
                if isinstance(layer, nn.Linear):
                    orthogonal_init(layer)       

    def forward(self, x, avail_a):
        logits = self.model(x)
        '''Mask the unavailable actions'''
        logits[avail_a == 0] = -1e10
        probs = torch.softmax(logits, dim=-1)
        return probs
    
    def transform_sample_data(self, list_sample_data, device):
        State = []
        for sample_data in list_sample_data:
            State.append(sample_data.state)
            
        tensor_state = torch.tensor(np.array(State)).to(device)
        return tensor_state

# 价值网络构造
class Critic(torch.nn.Module):
    def __init__(self, critic_input_dim, hid_shape, use_orthogonal_init = True):
        super(Critic, self).__init__()
        layers = []
        '''中心化的动作价值函数:所有智能体要同时给出自己的观测和相应的动作'''
        layer_shape = [critic_input_dim] + list(hid_shape) + [1]
        '''设置激活函数为 ReLU '''
        activation = nn.ReLU
        '''Build networks with For loop'''
        for j in range(len(layer_shape)-1):
            if j < len(layer_shape) - 2: 
                layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), activation()]
            else: 
                layers += [nn.Linear(layer_shape[j], layer_shape[j+1])]
        self.Q = nn.Sequential(*layers)   
        ''' 设置正交初始化 '''
        if use_orthogonal_init:
            for layer in self.Q:
                if isinstance(layer, nn.Linear):
                    orthogonal_init(layer)       

    def forward(self, x):
        return self.Q(x)
    
    def transform_sample_data(self, list_sample_data, device):
        State = []
        for sample_data in list_sample_data:
            State.append(sample_data.state)
            
        tensor_state = torch.tensor(np.array(State)).to(device)
        return tensor_state
    

class Agent:
    def __init__(self,agent_id):
        torch.manual_seed(0)
        self.state_dim = Config.state_dim
        self.obs_dim = Config.obs_dim_n[agent_id]
        self.action_dim = Config.action_dim_n[agent_id]
        self.agent_num = Config.agent_num
        self.actor_hidden_layers = Config.actor_hidden_layers
        self.critic_hidden_layers = Config.critic_hidden_layers
        self.actor_learning_rate = Config.actor_learning_rate
        self.critic_learning_rate = Config.critic_learning_rate
        self.use_agent_specific = Config.use_agent_specific
        self.actor_input_dim = self.obs_dim
        self.critic_input_dim = self.state_dim
        if self.use_agent_specific:
            print("------Critic模型输入中增加智能体特有的观测------")
            self.critic_input_dim += self.obs_dim
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        '''Build Actor and Critic'''
        self.actor = Actor(self.actor_input_dim,self.actor_hidden_layers,self.action_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr = self.actor_learning_rate)
        self.critic = Critic(self.critic_input_dim,self.critic_hidden_layers).to(self.device)
        self.critic_optimizer = torch.optim.Adam(params=self.critic.parameters(), lr = self.critic_learning_rate)
        self.model = [self.actor,self.critic]
        self.optimizer = [self.actor_optimizer,self.critic_optimizer]
        self.algo = Algo(model = self.model, config = Config, optimizer = self.optimizer, device = self.device)

    def take_action(self,obs,avail_a):
        with torch.no_grad():
            obs = torch.tensor(obs).view(1, self.obs_dim).to(self.device)
            avail_a = torch.tensor(avail_a).view(1, self.action_dim).to(self.device)
            probs = self.actor(obs, avail_a)
            action_dist = Categorical(probs=probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return [action.item(), log_prob.item()]
    
    def get_value(self,state,obs):
        with torch.no_grad():    
            if self.use_agent_specific:  # Add local obs of the agent
                critic_input = torch.tensor(np.concatenate([state, obs]), dtype=torch.float32)
            else:
                critic_input = torch.tensor(state, dtype=torch.float32)
            value = self.critic(critic_input)
            return value.cpu().numpy().item()

    def update(self,Long_Traj):
        actor_loss, critic_loss = self.algo.learn(Long_Traj)
        return actor_loss, critic_loss
    
    def best_action(self,obs,avail_a):
        with torch.no_grad():
            obs = torch.tensor(obs).view(1, self.obs_dim).to(self.device)
            avail_a = torch.tensor(avail_a).view(1, self.action_dim).to(self.device)
            action = np.argmax(self.actor(obs, avail_a).cpu().numpy())
        return action
    

    
