from base_algo import BaseAlgo
import torch
import numpy as np
import torch.nn.functional as F

class Algo(BaseAlgo):
    def __init__(self, model, config, optimizer=None, device=None, logger=None, monitor=None): 
        self.config = config
        self.tau = config.tau                             
        self.gamma = config.gamma                        
        self.batch_size = config.batch_size
        self.device = device
        self.Actor = model[0].to(self.device)
        self.Actor_Target = model[1].to(self.device)
        self.Critic = model[2].to(self.device)
        self.Critic_Target = model[3].to(self.device)
        self.actor_optimizer = optimizer[0]
        self.critic_optimizer = optimizer[1]
        self.train_step = 0

    def learn(self, list_sample_data):
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
            next_value = self.Critic_Target(next_state, self.Actor_Target(next_state))
            '''dw(dead and win) for TD_target'''
            TD_target = reward + self.gamma * next_value * (1 - dw)

        '''critic loss'''
        critic_loss = F.mse_loss(self.Critic(state,action),TD_target)

        self.critic_optimizer.zero_grad()           # critic梯度清零
        critic_loss.backward()						# critic反向传播
        self.critic_optimizer.step()                # 更新critic模型参数

        '''actor loss'''
        actor_loss = - self.Critic(state, self.Actor(state))

        self.actor_optimizer.zero_grad()			# actor梯度清零
        actor_loss.mean().backward()                # actor反向传播
        self.actor_optimizer.step()					# 更新actor模型参数

        '''Update the frozen target models'''
        with torch.no_grad():
            for param, target_param in zip(self.Actor.parameters(), self.Actor_Target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.Critic.parameters(), self.Critic_Target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.train_step += 1						# 更新计数器
        return actor_loss.mean().detach().cpu().numpy() , critic_loss.detach().cpu().numpy() 

    
    def sample_data_check(self,list_sample_data):
        if not isinstance(list_sample_data, list):
            raise TypeError("Expected data to be a list")  # 数据类型错误 数据需要是一个列表！
        if len(list_sample_data) == 0:
            raise ValueError("Data cannot be empty")  # 数据为空错误