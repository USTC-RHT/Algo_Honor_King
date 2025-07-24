from base_algo import BaseAlgo
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Algo(BaseAlgo):
    def __init__(self, model, config, optimizer=None, device=None, logger=None, monitor=None): 
        self.config = config
        self.tau = config.tau                             
        self.gamma = config.gamma                        
        self.batch_size = config.batch_size
        self.clip_grad_max_norm = config.clip_grad_max_norm
        self.device = device
        self.Actor = model[0].to(self.device)
        self.Actor_Target = model[1].to(self.device)
        self.Critic = model[2].to(self.device)
        self.Critic_Target = model[3].to(self.device)
        self.actor_optimizer = optimizer[0]
        self.critic_optimizer = optimizer[1]
        self.train_step = 0

    def learn(self, list_sample_data, agent_n, agent_names, agent_name):
        '''
        - list_sample_data是从回放池中采样得到的样本集合(batch):{(obs,a,r,obs',dw)} -> 以列表形式
        且其中每个元素都是字典类型,字典的key是智能体的名称
        - agent_n是所有智能体的列表
        - agent_name是当前训练的智能体的名称
        '''
        self.sample_data_check(list_sample_data)    # 检查样本字段是否符合要求

        obs_n, action_n, reward_n, next_obs_n, dw_n =\
            zip(*[(sample_data.obs, sample_data.action, sample_data.reward, 
                sample_data.next_obs, sample_data.dw) for sample_data in list_sample_data])

        obs_all = []
        a_all = []
        infer_a_all = []
        next_obs_all = []
        next_a_all = []
        for i, name in enumerate(agent_names):
            # 提取名字为 name的 agent, batch_size 个样本的 obs
            obs_list = [dict[name] for dict in obs_n]
            a_list = [dict[name] for dict in action_n]
            next_obs_list = [dict[name] for dict in next_obs_n]

            # 转换为 [batch_size, obs_dim] 的 tensor
            obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32).to(self.device)
            a_tensor = torch.tensor(np.array(a_list), dtype=torch.float32).to(self.device)
            next_obs_tensor = torch.tensor(np.array(next_obs_list), dtype=torch.float32).to(self.device)            
            infer_a_all.append(agent_n[i].actor(obs_tensor))
            obs_all.append(obs_tensor)
            a_all.append(a_tensor)
            next_obs_all.append(next_obs_tensor)

            with torch.no_grad():
                '''列表的每一个元素'''
                next_a = agent_n[i].actor_target(next_obs_tensor)
            next_a_all.append(next_a)
        
        reward_list = [reward[agent_name] for reward in reward_n]
        dw_list = [dw[agent_name] for dw in dw_n]
        reward_tensor = torch.tensor(np.array(reward_list), dtype=torch.float32).unsqueeze(1).to(self.device)
        dw_tensor = torch.tensor(np.array(dw_list), dtype=torch.float32).unsqueeze(1).to(self.device)

        obs_all = torch.cat(obs_all, dim = 1)
        a_all = torch.cat(a_all, dim = 1)
        infer_a_all = torch.cat(infer_a_all, dim = 1)
        next_obs_all = torch.cat(next_obs_all, dim = 1)
        next_a_all = torch.cat(next_a_all, dim = 1)
        next_value = self.Critic_Target(next_obs_all, next_a_all)
        '''dw(dead and win) for TD_target'''
        TD_target = reward_tensor + self.gamma * next_value * (1 - dw_tensor)

        '''actor loss'''
        actor_loss = - self.Critic(obs_all,infer_a_all)

        '''critic loss'''
        critic_loss = F.mse_loss(self.Critic(obs_all,a_all),TD_target)

        self.actor_optimizer.zero_grad()			# actor梯度清零
        actor_loss.mean().backward()                # actor反向传播
        nn.utils.clip_grad_norm_(self.Actor.parameters(), self.clip_grad_max_norm)   # actor梯度裁剪
        self.actor_optimizer.step()					# 更新actor模型参数

        self.critic_optimizer.zero_grad()           # critic梯度清零
        critic_loss.backward()						# critic反向传播
        nn.utils.clip_grad_norm_(self.Critic.parameters(), self.clip_grad_max_norm)   # actor梯度裁剪
        self.critic_optimizer.step()                # 更新critic模型参数

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