from base_algo import BaseAlgo
import torch
import torch.nn.functional as F
import numpy as np

class Algo(BaseAlgo):
    def __init__(self, model, config, optimizer=None, device=None, logger=None, monitor=None): 
        # 离散表格型强化学习算法 不需要神经网络与优化器
        self.config = config                             # 配置文件
        self.gamma = config.gamma                        # 折扣因子
        self.device = device
        self.optimizer = optimizer
        # 模型 model代表拟合动作值Q的神经网络，在智能体中定义
        self.Q_main = model[0].to(self.device)           # 主神经网络
        self.Q_target = model[1].to(self.device)         # 目标神经网络                          
        self.train_step = 0

    def learn(self, list_sample_data):
        """
        Deep Q-learning(DQN)
        - list_sample_data是从回放池中采样得到的样本集合(batch):{(s,a,r,s')} -> 以列表形式
        - 用目标网络(target network)计算目标值: target_q_value = r + gamma * max q(s',a',w_T)
        其中：
        r 表示R(s,a),即在状态s下采取动作a所获得的奖励
        gamma 是折扣因子, 用于平衡当前奖励和未来奖励的重要性
        max q(s',a',w_T) 表示在新状态s'下采取所有可能动作a'的最大Q值,用目标网络w_T计算
        """
        self.sample_data_check(list_sample_data)    # 检查样本字段是否符合要求

        # model_input_data是一个形状为(batch_size, state_dim)的 tensor
        model_input_data = self.Q_main.transform_sample_data(list_sample_data,self.device)	    # 将样本转换为模型可推理的格式

        # model_output_data是一个形状为(batch_size, action_dim)的 tensor,是主神经网络的输出
        model_output_data = self.Q_main.forward(model_input_data)				    # 模型推理
        self.optimizer.zero_grad()					                                # 清空梯度
        loss = self.calculate_loss(list_sample_data, model_output_data)	            # 计算loss
        loss.backward()													            # 计算梯度
        self.optimizer.step()						                                # 更新模型

        if self.train_step % self.config.target_network_update_freq == 0:
            self.Q_target.load_state_dict(self.Q_main.state_dict())                 # 更新目标网络
        self.train_step += 1                                                        # 更新计数器
    
    def calculate_loss(self,list_sample_data, model_output_data):
        loss = 0

        actions, rewards, next_states, dones = zip(*[(sample_data.action, sample_data.reward, sample_data.next_state,
                                                      int(sample_data.done)) for sample_data in list_sample_data])
        rewards = torch.tensor(rewards).to(self.device)
        dones = torch.tensor(dones).to(self.device)
        
        main_q_values = model_output_data[np.arange(len(actions)),actions]
        next_state_tensor = torch.tensor(next_states).to(self.device)
        max_q_values = torch.max(self.Q_target.forward(next_state_tensor).detach(), dim=1).values
        target_q_values = rewards + self.gamma * max_q_values * (1 - dones)
        loss += F.mse_loss(main_q_values,target_q_values)
        return loss
    
    def sample_data_check(self,list_sample_data):
        if not isinstance(list_sample_data, list):
            raise TypeError("Expected data to be a list")  # 数据类型错误 数据需要是一个列表！
        if len(list_sample_data) == 0:
            raise ValueError("Data cannot be empty")  # 数据为空错误
        