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
        self.action_dim = config.action_dim
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
        self.train_step += 1						# 更新计数器
        self.sample_data_check(list_sample_data)    # 检查样本字段是否符合要求

        states, actions, rewards, next_states, dws =\
            zip(*[(sample_data.state, sample_data.action, sample_data.reward, 
                sample_data.next_state, sample_data.dw) for sample_data in list_sample_data])

        state = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)         
        action = torch.tensor(np.array(actions), dtype=torch.int64, device=self.device).unsqueeze(1)     
        reward = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dw = torch.tensor(np.array(dws), dtype=torch.int, device=self.device).unsqueeze(1)

        with torch.no_grad():
            ''' 以下tensor尺寸均为[batch_size, action_dim] '''
            next_action_dist = self.Actor(next_state)           
            next_log_probs = torch.log(next_action_dist + 1e-8)    
            target_Q1, target_Q2 = self.Critic_Target(next_state)
            min_Q = torch.min(target_Q1, target_Q2)
            next_state_value =  (next_action_dist * (min_Q - self.log_alpha.exp() * next_log_probs)).sum(dim = 1, keepdim=True)
            ''' Clipped Double Q-learning '''
            TD_target = reward + self.gamma * next_state_value * (1 - dw)

        ''' Critic loss '''
        Q1, Q2 = self.Critic(state)
        Q1, Q2 = Q1.gather(1,action), Q2.gather(1,action)
        critic_loss = F.mse_loss(Q1,TD_target) + F.mse_loss(Q2,TD_target)

        ''' Optimize the critic '''
        self.critic_optimizer.zero_grad()           # critic梯度清零
        critic_loss.backward()				        # critic反向传播
        self.critic_optimizer.step()                # 更新critic模型参数

        ''' Freeze critic so you don't waste computational effort computing gradients for them when update actor '''
        for params in self.Critic.parameters(): params.requires_grad = False
        '''actor loss'''
        action_dist = self.Actor(state)
        log_probs = torch.log(action_dist + 1e-8)
        current_Q1, current_Q2 = self.Critic(state)
        min_Q = torch.min(current_Q1, current_Q2)
        actor_loss = action_dist * (self.log_alpha.exp() * log_probs - min_Q)
        actor_loss = actor_loss.sum(dim = 1)

        ''' Optimize the actor '''
        self.actor_optimizer.zero_grad()			# actor梯度清零
        actor_loss.mean().backward()                # actor反向传播
        self.actor_optimizer.step()					# 更新actor模型参数

        for params in self.Critic.parameters(): params.requires_grad = True

        if self.adaptive_alpha:
            log_alpha_loss = - self.log_alpha.exp() * (log_probs.detach() + self.target_entropy)
            log_alpha_loss = (action_dist.detach() * log_alpha_loss).sum(dim = 1)
        
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