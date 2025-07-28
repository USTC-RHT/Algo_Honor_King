from base_algo import BaseAlgo
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F

class Algo(BaseAlgo):
    def __init__(self, model, config, optimizer=None, device=None, logger=None, monitor=None): 
        self.config = config
        self.tau = config.tau                             
        self.gamma = config.gamma                        
        self.batch_size = config.batch_size
        self.use_grad_clip = config.use_grad_clip
        self.clip_grad_max_norm = config.clip_grad_max_norm
        self.device = device
        self.Actor = model[0].to(self.device)
        self.Actor_Target = model[1].to(self.device)
        self.Critic = model[2].to(self.device)
        self.Critic_Target = model[3].to(self.device)
        self.actor_optimizer = optimizer[0]
        self.critic_optimizer = optimizer[1]
        self.train_step = 0

    def learn(self, list_sample_data, agent_n, agent_id):
        '''
        - list_sample_data是从回放池中采样得到的样本集合(batch):
        {(obs_n, action_n, reward_n, next_obs_n, dw_n)} -> 以列表形式
        且其中每个元素都是tensor类型,例如:obs_n[i]是维度为[batch_size, obs_dim]的tensor
        - agent_n是所有智能体的列表
        - agent_id是当前训练的智能体的id
        '''
        agent_num = len(agent_n)
        self.sample_data_check(list_sample_data)   
        obs_n, action_n, reward_n, next_obs_n, dw_n = list_sample_data

        next_a_all = []
        for i in range(agent_num):
            with torch.no_grad():
                next_a = agent_n[i].actor_target(next_obs_n[i])
                next_a_all.append(next_a)

        ''' 按照当前智能体的'agent_id'再选择动作并保持其他智能体的动作不变 '''
        ''' 注意此处需要进行深拷贝 避免共享引用与in-place修改'''
        infer_a_all = copy.deepcopy(action_n)
        infer_a_all[agent_id] = agent_n[agent_id].actor(obs_n[agent_id])

        obs_all = torch.cat(obs_n, dim = 1).to(self.device)
        a_all = torch.cat(action_n, dim = 1).to(self.device)
        infer_a_all = torch.cat(infer_a_all, dim = 1).to(self.device)
        next_obs_all = torch.cat(next_obs_n, dim = 1).to(self.device)
        next_a_all = torch.cat(next_a_all, dim = 1).to(self.device)
        reward_tensor = reward_n[agent_id]
        dw_tensor = dw_n[agent_id]

        with torch.no_grad():
            next_value = self.Critic_Target(next_obs_all, next_a_all)
            '''dw(dead and win) for TD_target'''
            TD_target = reward_tensor + self.gamma * next_value * (1 - dw_tensor)

        '''critic loss'''
        critic_loss = F.mse_loss(self.Critic(obs_all,a_all),TD_target)

        self.critic_optimizer.zero_grad()           # critic梯度清零
        critic_loss.backward()						# critic反向传播
        if self.use_grad_clip:
            nn.utils.clip_grad_norm_(self.Critic.parameters(), self.clip_grad_max_norm)   # actor梯度裁剪
        self.critic_optimizer.step()                # 更新critic模型参数

        '''actor loss'''
        actor_loss = - self.Critic(obs_all,infer_a_all)

        self.actor_optimizer.zero_grad()			# actor梯度清零
        actor_loss.mean().backward()                # actor反向传播
        if self.use_grad_clip:
            nn.utils.clip_grad_norm_(self.Actor.parameters(), self.clip_grad_max_norm)   # actor梯度裁剪
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
        if len(list_sample_data) == 0:
            raise ValueError("Data cannot be empty")  # 数据为空错误