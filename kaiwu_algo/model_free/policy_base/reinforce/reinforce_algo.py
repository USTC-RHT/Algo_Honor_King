from base_algo import BaseAlgo
import torch

class Algo(BaseAlgo):
    def __init__(self, model, config, optimizer=None, device=None, logger=None, monitor=None): 
        self.config = config                             # 配置文件
        self.gamma = config.gamma                        # 折扣因子
        self.Policy_Net = model                          # 在这里模型 model表示策略神经网络，在智能体中定义
        self.optimizer = optimizer
        self.device = device
        self.model = model.to(self.device)
        self.train_step = 0

    def learn(self, list_sample_data):
        """
        Policy Gradient by Monte Carlo(REINFORCE)
        - list_sample_data是一个episode列表:[(s_0,a_0,r_1),...,(s_{T-1},a_{T-1},r_T)]
        使用以下公式计算返回值：
        - Return = r_{t+1} + gamma * r_{t+2} + ... + gamma^(T-t-1) * r_T
        """
        self.sample_data_check(list_sample_data)    # 检查样本字段是否符合要求

        # model_input_data是一个形状为(batch_size, state_dim)的 tensor
        model_input_data = self.model.transform_sample_data(list_sample_data,self.device)	    # 将样本转换为模型可推理的格式

        # model_output_data是一个形状为(batch_size, action_dim)的 tensor
        model_output_data = self.model.forward(model_input_data)				    # 模型推理
        self.optimizer.zero_grad()					                                # 清空梯度
        loss = self.calculate_loss(list_sample_data, model_output_data)	            # 计算loss
        loss.backward()													            # 计算梯度
        self.optimizer.step()						                                # 更新模型
        self.train_step += 1						                                # 更新计数器
    
    def calculate_loss(self,list_sample_data, model_output_data):
        Return, loss = 0, 0
        i = -1        

        # 计算每个状态-动作对的返回值  倒序累加利于计算
        for sample_data in reversed(list_sample_data):
            Return = self.gamma * Return + sample_data.reward
            log_prob = torch.log(model_output_data[i,sample_data.action])
            loss += -log_prob * Return  # 每一步的损失函数
            i = i - 1

        return loss
    
    def sample_data_check(self,list_sample_data):
        if not isinstance(list_sample_data, list):
            raise TypeError("Expected data to be a list")  # 数据类型错误 数据需要是一个列表！
        if len(list_sample_data) == 0:
            raise ValueError("Data cannot be empty")  # 数据为空错误