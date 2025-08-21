from base_algo import BaseAlgo
import torch
import copy
import numpy as np
from torch import nn
from torch.utils.data.sampler import *
from torch.distributions import Categorical

class Algo(BaseAlgo):
    def __init__(self, model, config, optimizer=None, device=None, logger=None, monitor=None): 
        self.config = config                             # 配置文件
        self.gamma = config.gamma                        
        self.lambd = config.lambd
        self.agent_num = config.agent_num                        
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.mini_batch_size = config.mini_batch_size
        self.clip_rate = config.clip_rate
        self.entropy_coef = config.entropy_coef
        self.clip_grad_max_norm = config.clip_grad_max_norm
        self.Advantage_Normal = config.Advantage_Normal
        self.use_rnn = config.use_rnn
        self.use_agent_specific = config.use_agent_specific
        self.device = device
        self.Actor = model[0].to(self.device)
        self.Critic = model[1].to(self.device)
        self.actor_optimizer = optimizer[0]
        self.critic_optimizer = optimizer[1]
        self.train_step = 0

    def learn(self, batch):
        '''所有agent共享一个actor网络和一个critic网络,适合'同质'多智能体环境.'''
        ''' Input: batch是一个字典 
        包含obs_n、state、v_n、avail_a_n、a_n、logprob_a_n、r、dw、active、max_episode_len
        batch = {'obs_n': np.zeros([self.batch_size, self.episode_limit, self.N, self.obs_dim]),
                'state': np.zeros([self.batch_size, self.episode_limit, self.state_dim]),
                'v_n': np.zeros([self.batch_size, self.episode_limit + 1, self.N]),
                'avail_a_n': np.ones([self.batch_size, self.episode_limit, self.N, self.action_dim]),
                'a_n': np.zeros([self.batch_size, self.episode_limit, self.N]),
                'logprob_a_n': np.zeros([self.batch_size, self.episode_limit, self.N]),
                'r': np.zeros([self.batch_size, self.episode_limit, self.N]),
                'dw': np.ones([self.batch_size, self.episode_limit, self.N]),  
                'active': np.zeros([self.batch_size, self.episode_limit, self.N])}
                
        obs_n: n个智能体的观测,用于各智能体的策略网络输入,执行时只需局部观测
        state: 全局状态,用于中心化值函数(Critic)输入
        v_n: 每个智能体每个时间步(含终止步 t+1)的价值估计
        avail_a_n: 每个智能体可用动作的mask
        a_n: 每个智能体实际采取的离散动作
        logprob_a_n: 每个智能体每步选到当前动作的对数概率
        r: 每个智能体每步获得的奖励
        dw: 每个智能体每步回合是否终止
        active: 每个智能体在该时间步是否存在/活跃
        max_episode_len: 该 batch 中轨迹的最大时间步长度

        Output: actor_loss, critic_loss'''

        max_episode_len = batch['max_episode_len']
        for key in batch.keys():
            if key != 'max_episode_len':
                batch[key] = batch[key].to(self.device)
        # Calculate the advantage using GAE
        A = []
        gae = 0
        with torch.no_grad():
            # TD_error.shape=(batch_size,max_episode_len,N)
            TD_error = batch['r'] + self.gamma * batch['v_n'][:, 1:] * (1 - batch['dw']) - batch['v_n'][:, :-1]
            for t in reversed(range(max_episode_len)):
                gae = TD_error[:, t] + self.gamma * self.lambd * gae
                A.insert(0, gae)
            A = torch.stack(A, dim=1)  # A.shape(batch_size,max_episode_len,N)
            TD_target = A + batch['v_n'][:, :-1]  # TD_target.shape(batch_size,max_episode_len,N)
            if self.Advantage_Normal:  # Trick 1: advantage normalization
                A_copy = copy.deepcopy(A.numpy())
                A_copy[batch['active'].numpy() == 0] = np.nan
                A = ((A - np.nanmean(A_copy)) / (np.nanstd(A_copy) + 1e-5))
                
        ''' PPO update '''
        ''' Optimize policy for K epochs '''
        for _ in range(self.num_epochs):
            for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
                '''actor loss'''
                if self.use_rnn:
                    # If use RNN, we need to reset the rnn_hidden of the actor and critic.
                    self.Actor.rnn_hidden = None
                    probs_now = []
                    for t in range(max_episode_len):
                        # prob.shape=(mini_batch_size*N, action_dim)
                        prob = self.Actor(batch['obs_n'][index, t].reshape(self.mini_batch_size * self.agent_num, -1),
                                          batch['avail_a_n'][index, t].reshape(self.mini_batch_size * self.agent_num, -1))
                        probs_now.append(prob.reshape(self.mini_batch_size, self.agent_num, -1))
                    # Stack them according to the time (dim=1)
                    new_prob = torch.stack(probs_now, dim=1)
                else:
                    new_prob = self.Actor(batch['obs_n'][index],batch['avail_a_n'][index])

                new_action_dist = Categorical(probs=new_prob)
                new_logprob_a = new_action_dist.log_prob(batch['a_n'][index])
                old_logprob_a = batch['logprob_a_n'][index]
                # a/b == exp(log(a)-log(b))
                ratio = torch.exp(new_logprob_a - old_logprob_a)

                surr1 = ratio * A[index]
                '''PPO-截断(PPO-Clip)'''
                surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * A[index]
                ppo_loss = -torch.min(surr1, surr2)

                entropy = new_action_dist.entropy()
                entropy_loss = - self.entropy_coef * entropy
                actor_loss = ppo_loss + entropy_loss
                actor_loss = (actor_loss * batch['active'][index]).sum() / batch['active'][index].sum()

                '''critic loss'''
                # 获取 agent的数量 N
                N = batch['obs_n'][index].shape[2]
                # 扩展 state 到 [B, T, 1, state_dim] 再 repeat 到 [B, T, N, state_dim]
                state = batch['state'][index]
                state = state.unsqueeze(2).repeat(1, 1, N, 1)

                '''use_agent_specific'''
                if self.use_agent_specific:
                    obs_n = batch['obs_n'][index]
                    critic_inputs = torch.cat([state, obs_n], dim=-1)
                else:
                    critic_inputs = state

                '''use_rnn'''
                if self.use_rnn:
                    # If use RNN, we need to reset the rnn_hidden of the actor and critic.
                    self.Critic.rnn_hidden = None
                    values_now = []
                    for t in range(max_episode_len):
                        # v.shape=(mini_batch_size*N,1)
                        v = self.Critic(critic_inputs[:, t].reshape(self.mini_batch_size * self.agent_num, -1))
                        values_now.append(v.reshape(self.mini_batch_size, self.agent_num))
                    values_now = torch.stack(values_now, dim=1).squeeze(dim=-1)
                else:
                    values_now = self.Critic(critic_inputs).squeeze(dim=-1)
                
                critic_loss = (values_now - TD_target[index]) ** 2
                critic_loss = (critic_loss * batch['active'][index]).sum() / batch['active'][index].sum()

                self.actor_optimizer.zero_grad()	 # actor梯度清零
                actor_loss.backward()                # actor反向传播
                nn.utils.clip_grad_norm_(self.Actor.parameters(), self.clip_grad_max_norm)   # actor梯度裁剪
                self.actor_optimizer.step()					# 更新actor模型参数

                self.critic_optimizer.zero_grad()           # critic梯度清零
                critic_loss.backward()						# critic反向传播
                nn.utils.clip_grad_norm_(self.Critic.parameters(), self.clip_grad_max_norm)   # critic梯度裁剪
                self.critic_optimizer.step()                # 更新critic模型参数

        self.train_step += 1		# 更新计数器
        return actor_loss.detach().cpu().numpy() , critic_loss.detach().cpu().numpy()