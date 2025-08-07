from base_algo import BaseAlgo
import torch
from torch import nn
from torch.utils.data.sampler import *

class Algo(BaseAlgo):
    def __init__(self, model, config, optimizer=None, device=None, logger=None, monitor=None): 
        self.config = config                             # 配置文件
        self.gamma = config.gamma
        self.tau = config.tau                            # 软更新系数
        self.agent_num = config.agent_num                        
        self.batch_size = config.batch_size
        self.use_rnn = config.use_rnn
        self.clip_grad_max_norm = config.clip_grad_max_norm
        self.device = device
        # 模型 model代表拟合动作值Q的神经网络，在智能体中定义
        self.Q_main = model[0].to(self.device)           # 主神经网络
        self.Q_target = model[1].to(self.device)         # 目标神经网络  
        self.optimizer = optimizer                        
        self.train_step = 0

    def learn(self, batch):
        '''所有agent共享一个actor网络和一个critic网络,适合'同质'多智能体环境.'''
        max_episode_len = batch['max_episode_len']
        for key in batch.keys():
            if key != 'max_episode_len':
                batch[key] = batch[key].to(self.device)

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

        self.optimizer.zero_grad()	         # 梯度清零
        loss.backward()                      # 反向传播
        nn.utils.clip_grad_norm_(self.Q_main.parameters(), self.clip_grad_max_norm)   # actor梯度裁剪
        self.optimizer.step()			     # 更新模型参数

        self.train_step += 1		# 更新计数器
        return loss.detach().cpu().numpy()