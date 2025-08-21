from base_algo import BaseAlgo
import torch
import numpy as np
import torch.nn.functional as F

class Algo(BaseAlgo):
    def __init__(self, model, config, optimizer=None, device=None, logger=None, monitor=None): 
        self.config = config
        self.device = device
        self.tau = config.tau                             
        self.gamma = config.gamma
        self.adaptive_alpha = config.adaptive_alpha
        self.target_entropy = config.target_entropy
        ''' model = [self.actor,self.critic,self.critic_target,self.log_alpha] '''
        self.Actor = model[0].to(self.device)
        self.Critic = model[1].to(self.device)
        self.Critic_Target = model[2].to(self.device)
        self.log_alpha = model[3].to(self.device)
        ''' optimizer = [self.actor_optimizer,self.critic_optimizer,self.log_alpha_optimizer] '''
        self.actor_optimizer = optimizer[0]
        self.critic_optimizer = optimizer[1]
        self.log_alpha_optimizer = optimizer[2]
        self.train_step = 0

    def learn(self, list_sample_data):
        """
        # Input: list_sample_data是从回放池中采样得到的样本集合(batch):{(s,a,r,s',dw)} -> 以列表形式

        # Output: actor_loss, critic_loss
        """
        self.train_step += 1						# 更新计数器
        self.sample_data_check(list_sample_data)    # 检查样本字段是否符合要求

        states, actions, rewards, next_states, dws =\
            zip(*[(sample_data.state, sample_data.action, sample_data.reward, 
                sample_data.next_state, sample_data.dw) for sample_data in list_sample_data])

        state = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)         
        action = torch.tensor(np.array(actions), dtype=torch.float32, device=self.device)     
        reward = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dw = torch.tensor(np.array(dws), dtype=torch.int, device=self.device).unsqueeze(1)

        with torch.no_grad():
            # Compute the target Q
            next_action, log_pi_a_next = self.Actor.sample_act(next_state)
            target_Q1, target_Q2 = self.Critic_Target(next_state, next_action)
            ''' Clipped Double Q-learning '''
            TD_target = reward + self.gamma * (torch.min(target_Q1, target_Q2) \
                                            - self.log_alpha.exp() * log_pi_a_next) * (1 - dw)

        ''' Critic loss '''
        Q1, Q2 = self.Critic(state,action)
        critic_loss = F.mse_loss(Q1,TD_target) + F.mse_loss(Q2,TD_target)

        ''' Optimize the critic '''
        self.critic_optimizer.zero_grad()           # critic梯度清零
        critic_loss.backward()						# critic反向传播
        self.critic_optimizer.step()                # 更新critic模型参数

        ''' Freeze critic so you don't waste computational effort computing gradients for them when update actor '''
        for params in self.Critic.parameters(): params.requires_grad = False
        '''actor loss'''
        rsample_a, log_pi_a = self.Actor.sample_act(state)
        current_Q1, current_Q2 = self.Critic(state, rsample_a)
        rsample_Q = torch.min(current_Q1, current_Q2)
        actor_loss = self.log_alpha.exp() * log_pi_a - rsample_Q

        ''' Optimize the actor '''
        self.actor_optimizer.zero_grad()			# actor梯度清零
        actor_loss.mean().backward()                # actor反向传播
        self.actor_optimizer.step()					# 更新actor模型参数

        for params in self.Critic.parameters(): params.requires_grad = True

        if self.adaptive_alpha:
            log_alpha_loss = - self.log_alpha.exp() * (log_pi_a + self.target_entropy).detach()
        
            ''' Optimize the alpha '''
            self.log_alpha_optimizer.zero_grad()        # log_alpha梯度清零
            log_alpha_loss.mean().backward()            # log_alpha反向传播
            self.log_alpha_optimizer.step()             # 更新log_alpha模型参数

        '''Update the frozen target models'''
        with torch.no_grad():
            for param, target_param in zip(self.Critic.parameters(), self.Critic_Target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return actor_loss.mean().detach().cpu().numpy() , critic_loss.detach().cpu().numpy() 

    def sample_data_check(self,list_sample_data):
        if not isinstance(list_sample_data, list):
            raise TypeError("Expected data to be a list")  # 数据类型错误 数据需要是一个列表！
        if len(list_sample_data) == 0:
            raise ValueError("Data cannot be empty")  # 数据为空错误