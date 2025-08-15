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
        self.QMIX_main = model[2].to(self.device)
        self.QMIX_target = model[3].to(self.device)
        self.optimizer = optimizer                        
        self.train_step = 0

    def learn(self, batch):
        '''实现中心化训练、去中心化执行:训练时可以用全局信息优化团队Q值,执行时每个体只需用自己的观察即可行动。
           所有agent共享一个动作价值网络,适合'同质'多智能体环境。
           QMIX: 用一个混合网络(Mixing Network)来实现 Q_total 的计算,
           混合网络的权重通过超网络(hypernetwork)结合全局信息生成,混合网络接收所有个体Q值为输入。'''
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
            for t in range(max_episode_len):  # t=0,1,2,...(episode_len-1)
                # q_main.shape=(batch_size * N,action_dim)
                q_main = self.Q_main(inputs[:, t].reshape(-1, input_dim))
                q_target = self.Q_target(inputs[:, t + 1].reshape(-1, input_dim))
                # q_mains.shape=(batch_size,N,action_dim)
                q_mains.append(q_main.reshape(self.batch_size, self.agent_num, -1))
                q_targets.append(q_target.reshape(self.batch_size, self.agent_num, -1))

            # Stack them according to the time (dim=1)
            # q_evals.shape=(batch_size,max_episode_len,N,action_dim)
            q_mains = torch.stack(q_mains, dim=1)
            q_targets = torch.stack(q_targets, dim=1)
        else:
            q_mains = self.Q_main(inputs[:, :-1])
            q_targets = self.Q_target(inputs[:, 1:])
        
        with torch.no_grad():
            if self.use_double_q:  # If use double q-learning, we use eval_net to choose actions,and use target_net to compute q_target
                q_main_last = self.Q_main(inputs[:, -1].reshape(-1, input_dim)).reshape(self.batch_size, 1, self.agent_num, -1)
                q_mains_next = torch.cat([q_mains[:, 1:], q_main_last], dim=1) # q_evals_next.shape=(batch_size,max_episode_len,N,action_dim)
                q_mains_next[batch['avail_a_n'][:, 1:] == 0] = -999999
                a_argmax = torch.argmax(q_mains_next, dim=-1, keepdim=True)  # a_max.shape=(batch_size,max_episode_len, N, 1)
                q_targets = torch.gather(q_targets, dim=-1, index=a_argmax).squeeze(-1)  # q_targets.shape=(batch_size, max_episode_len, N)
            else:
                q_targets[batch['avail_a_n'][:, 1:] == 0] = -999999
                q_targets = q_targets.max(dim=-1)[0]  # q_targets.shape=(batch_size, max_episode_len, N)

        # batch['a_n'].shape(batch_size,max_episode_len, N)
        q_mains = torch.gather(q_mains, dim=-1, index=batch['a_n'].unsqueeze(-1)).squeeze(-1)  # q_evals.shape(batch_size, max_episode_len, N)

        q_total_main = self.QMIX_main(q_mains,batch['state'][:, :-1])
        q_total_target = self.QMIX_target(q_targets,batch['state'][:, 1:])
        TD_target = batch['r'] + self.gamma * (1 - batch['dw']) * q_total_target

        TD_error = (q_total_main - TD_target)
        mask_td_error = TD_error * batch['active']
        loss = (mask_td_error ** 2).sum() / batch['active'].sum()
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            parameters = list(self.QMIX_main.parameters()) + list(self.Q_main.parameters())
            torch.nn.utils.clip_grad_norm_(parameters, self.clip_grad_max_norm)
        self.optimizer.step()

        if self.use_hard_update:
            # hard update
            if self.train_step % self.target_network_update_freq == 0:
                self.Q_target.load_state_dict(self.Q_main.state_dict())
                self.QMIX_target.load_state_dict(self.QMIX_main.state_dict())
        else:
            # Softly update the target networks
            for param, target_param in zip(self.Q_main.parameters(), self.Q_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.QMIX_main.parameters(), self.QMIX_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.train_step += 1		# 更新计数器
        return loss.detach().cpu().numpy()