from base_algo import BaseAlgo
import torch
import copy
import math
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Algo(BaseAlgo):
    def __init__(self, model, config, optimizer=None, device=None, logger=None, monitor=None): 
        self.config = config                             # 配置文件
        self.gamma = config.gamma                        
        self.L2_reg = config.L2_reg
        self.entropy_coef = config.entropy_coef
        self.entropy_coef_decay = config.entropy_coef_decay
        self.clip_grad_max_norm = config.clip_grad_max_norm
        self.Advantage_Normal = config.Advantage_Normal
        self.device = device
        self.Actor = model[0].to(self.device)
        self.Critic = model[1].to(self.device)
        self.actor_optimizer = optimizer[0]
        self.critic_optimizer = optimizer[1]
        self.train_step = 0

    def learn(self, list_sample_data):
        self.sample_data_check(list_sample_data)    # 检查样本字段是否符合要求

        states, actions, rewards, next_states, dones =\
            zip(*[(sample_data.state, sample_data.action, sample_data.reward, 
                sample_data.next_state,int(sample_data.done)) for sample_data in list_sample_data])

        state = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)         
        action = torch.tensor(np.array(actions), dtype=torch.int64, device=self.device).unsqueeze(1)     
        reward = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        done = torch.tensor(np.array(dones), dtype=torch.int64, device=self.device).unsqueeze(1)

        with torch.no_grad():
            value = self.Critic(state)
            next_value = self.Critic(next_state)
            TD_target = reward + self.gamma * next_value * (1 - done)
            A = TD_target - value

            if self.Advantage_Normal:
                A= (A - A.mean()) / ((A.std() + 1e-4))  #sometimes helps 

        self.entropy_coef *= self.entropy_coef_decay     # exploring decay 探索衰减

        '''actor loss'''
        probs = self.Actor(state)
        log_probs = torch.log(probs.gather(1, action))

        a2c_loss = torch.mean(-log_probs * A.detach())

        entropy = Categorical(probs = probs).entropy()
        entropy_loss = - self.entropy_coef * entropy
        actor_loss = a2c_loss + entropy_loss

        '''critic loss'''
        critic_loss = F.mse_loss(self.Critic(state),TD_target)

        '''L2正则化 在损失函数中添加一个正则项来防止过拟合'''
        for name, param in self.Critic.named_parameters():
            if 'weight' in name:
                critic_loss += self.L2_reg * param.pow(2).sum()

        self.actor_optimizer.zero_grad()			# actor梯度清零
        actor_loss.mean().backward()                # actor反向传播
        ''' 
        actor梯度裁剪
        nn.utils.clip_grad_norm_(self.Actor.parameters(), self.clip_grad_max_norm)
        '''
        self.actor_optimizer.step()					# 更新actor模型参数

        self.critic_optimizer.zero_grad()           # critic梯度清零
        critic_loss.backward()						# critic反向传播
        self.critic_optimizer.step()                # 更新critic模型参数

        self.train_step += 1						# 更新计数器
        return actor_loss.mean().detach().cpu().numpy() , critic_loss.detach().cpu().numpy() 

    
    def sample_data_check(self,list_sample_data):
        if not isinstance(list_sample_data, list):
            raise TypeError("Expected data to be a list")  # 数据类型错误 数据需要是一个列表！
        if len(list_sample_data) == 0:
            raise ValueError("Data cannot be empty")  # 数据为空错误