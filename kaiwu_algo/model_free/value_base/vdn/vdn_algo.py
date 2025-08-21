from base_algo import BaseAlgo
import torch
from torch.utils.data.sampler import *

class Algo(BaseAlgo):
    def __init__(self, model, config, optimizer=None, device=None, logger=None, monitor=None): 
        self.config = config                             # 配置文件
        self.gamma = config.gamma
        self.tau = config.tau                            # 软更新系数
        self.agent_num = config.agent_num
        self.batch_size = config.batch_size                        
        self.use_rnn = config.use_rnn
        self.use_double_q = config.use_double_q
        self.add_agent_id = config.add_agent_id
        self.add_last_action = config.add_last_action
        self.use_hard_update = config.use_hard_update
        self.use_grad_clip = config.use_grad_clip
        self.clip_grad_max_norm = config.clip_grad_max_norm
        self.target_network_update_freq = config.target_network_update_freq
        self.device = device
        self.Q_main = model[0].to(self.device)           # 主神经网络
        self.Q_target = model[1].to(self.device)         # 目标神经网络  
        self.optimizer = optimizer                        
        self.train_step = 0

    def learn(self, batch):
        '''实现中心化训练、去中心化执行:训练时可以用全局信息优化团队Q值,执行时每个体只需用自己的观察即可行动。
           所有agent共享一个动作价值网络,适合'同质'多智能体环境。
           VDN: 将团队的全局Q值直接线性分解为各个智能体的Q值之和。'''
        ''' Input: batch是一个字典 
        包含obs_n、state、avail_a_n、a_n、last_onehot_a_n、r、dw、active、max_episode_len
        buffer = {'obs_n': np.zeros([self.buffer_size, self.episode_limit + 1, self.N, self.obs_dim]),
                'avail_a_n': np.ones([self.buffer_size, self.episode_limit + 1, self.N, self.action_dim]),
                'a_n': np.zeros([self.buffer_size, self.episode_limit, self.N]),
                'last_onehot_a_n': np.zeros([self.buffer_size, self.episode_limit + 1, self.N, self.action_dim]),
                'r': np.zeros([self.buffer_size, self.episode_limit, 1]),
                'dw': np.ones([self.buffer_size, self.episode_limit, 1]),
                'active': np.zeros([self.buffer_size, self.episode_limit, 1])
                }
        
        obs_n: n个智能体的观测,用于各智能体的策略网络输入,执行时只需局部观测
        avail_a_n: 每个智能体可用动作的mask
        a_n: 每个智能体实际采取的离散动作
        last_onehot_a_n: 每个智能体上一时间步的onehot动作
        r: 每个智能体每步获得的奖励
        dw: 每个智能体每步回合是否终止
        active: 每个智能体在该时间步是否存在/活跃
        max_episode_len: 该 batch 中轨迹的最大时间步长度

        Output: 动作价值网络的loss'''
        max_episode_len = batch['max_episode_len']
        for key in batch.keys():
            if key != 'max_episode_len':
                batch[key] = batch[key].to(self.device)

        inputs = [batch['obs_n']]
        if self.add_last_action:
            inputs.append(batch['last_onehot_a_n'])
        if self.add_agent_id:
            agent_id = torch.eye(self.agent_num).unsqueeze(0).unsqueeze(0)  # [1, 1, agent_num, agent_num]
            agent_id_one_hot = agent_id.expand(self.batch_size, max_episode_len + 1, -1, -1)
            inputs.append(agent_id_one_hot)

        # inputs.shape = (bach_size,max_episode_len + 1,N,input_dim)
        inputs = torch.cat([x for x in inputs], dim=-1)
        input_dim = inputs.shape[3]

        '''use_rnn'''
        if self.use_rnn:
            # If use RNN, we need to reset the rnn_hidden of the actor and critic.
            self.Q_main.rnn_hidden = None
            self.Q_target.rnn_hidden = None
            q_mains, q_targets = [], []
            # t=0,1,2,...(episode_len-1)
            for t in range(max_episode_len):  
                # q_main.shape=(batch_size * N,action_dim)
                q_main = self.Q_main(inputs[:, t].reshape(-1, input_dim))
                q_target = self.Q_target(inputs[:, t + 1].reshape(-1, input_dim))
                # q_mains.shape=(batch_size,N,action_dim)
                q_mains.append(q_main.reshape(self.batch_size, self.agent_num, -1))
                q_targets.append(q_target.reshape(self.batch_size, self.agent_num, -1))

            # Stack them according to the time (dim=1)
            # q_mains.shape=(batch_size,max_episode_len,N,action_dim)
            q_mains = torch.stack(q_mains, dim=1)
            q_targets = torch.stack(q_targets, dim=1)
        else:
            q_mains = self.Q_main(inputs[:, :-1])
            q_targets = self.Q_target(inputs[:, 1:])
        
        with torch.no_grad():
            # If use double q-learning, we use main_net to choose actions,and use target_net to compute q_target
            if self.use_double_q: 
                q_main_last = self.Q_main(inputs[:, -1].reshape(-1, input_dim)).reshape(self.batch_size, 1, self.agent_num, -1)
                # q_mains_next.shape=(batch_size,max_episode_len,N,action_dim)
                q_mains_next = torch.cat([q_mains[:, 1:], q_main_last], dim=1) 
                q_mains_next[batch['avail_a_n'][:, 1:] == 0] = -999999
                # a_argmax.shape=(batch_size,max_episode_len, N, 1)
                a_argmax = torch.argmax(q_mains_next, dim=-1, keepdim=True)  
                # q_targets.shape=(batch_size, max_episode_len, N)
                q_targets = torch.gather(q_targets, dim=-1, index=a_argmax).squeeze(-1)  
            else:
                q_targets[batch['avail_a_n'][:, 1:] == 0] = -999999
                # q_targets.shape=(batch_size, max_episode_len, N)
                q_targets = q_targets.max(dim=-1)[0]

        # batch['a_n'].shape(batch_size,max_episode_len, N)
        # q_mains.shape(batch_size, max_episode_len, N)
        q_mains = torch.gather(q_mains, dim=-1, index=batch['a_n'].unsqueeze(-1)).squeeze(-1)


        q_total_main = torch.sum(q_mains, dim=-1, keepdim=True)
        q_total_target = torch.sum(q_targets, dim=-1, keepdim=True)
        # targets.shape=(batch_size,max_episode_len,1)
        TD_target = batch['r'] + self.gamma * (1 - batch['dw']) * q_total_target

        TD_error = (q_total_main - TD_target)
        mask_td_error = TD_error * batch['active']
        loss = (mask_td_error ** 2).sum() / batch['active'].sum()
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.Q_main.parameters(), self.clip_grad_max_norm)
        self.optimizer.step()

        if self.use_hard_update:
            # hard update
            if self.train_step % self.target_network_update_freq == 0:
                self.Q_target.load_state_dict(self.Q_main.state_dict())
        else:
            # Softly update the target networks
            for param, target_param in zip(self.Q_main.parameters(), self.Q_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.train_step += 1		# 更新计数器
        return loss.detach().cpu().numpy()