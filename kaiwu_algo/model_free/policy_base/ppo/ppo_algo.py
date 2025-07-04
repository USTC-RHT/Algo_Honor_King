from base_algo import BaseAlgo
import torch
import copy
import numpy as np

class Algo(BaseAlgo):
    def __init__(self, model, config, optimizer=None, device=None, logger=None, monitor=None): 
        self.config = config                             # 配置文件
        self.gamma = config.gamma                        # 折扣因子
        self.lambd = config.lambd                        # GAE平滑系数
        self.device = device
        self.Actor = model[0].to(self.device)            # 在这里模型 model表示策略神经网络，在智能体中定义
        self.Critic = model[1].to(self.device)
        self.actor_optimizer = optimizer[0]
        self.critic_optimizer = optimizer[1]
        self.train_step = 0

    def learn(self, list_sample_data):
        """
        智能体连续地与环境交互,收集到一定步数的长轨迹才停止
        - list_sample_data是一个长轨迹(Long Trajectory):
        [(s_0,a_0,r_1,s_1,logprob_a_0,dw_1,done_1),...,(s_{T-1},a_{T-1},r_T,s_T,logprob_a_{T-1},dw_T,done_T)]
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

        states, actions, rewards, next_states, logprob_actions, dws, dones = zip(*[(sample_data.state, sample_data.action, 
                                                    sample_data.reward, sample_data.next_state,sample_data.logprob_a, sample_data.dw,
                                                    int(sample_data.done)) for sample_data in list_sample_data])

        state = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)         
        action = torch.tensor(np.array(actions), dtype=torch.int, device=self.device)     
        reward = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)       
        next_state = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        logprob_a = torch.tensor(np.array(logprob_actions), dtype=torch.float32, device=self.device)
        dw = torch.tensor(np.array(dws), dtype=torch.int, device=self.device)  
        done = torch.tensor(np.array(dones), dtype=torch.int, device=self.device)  

        ''' Use TD + GAE + LongTrajectory to compute Advantage and TD target'''
        with torch.no_grad():
            value = self.Critic(next_state)
            '''dw(dead and win) for TD_target and Adv'''
            td_target = reward + self.gamma * value * (1 - dw)
            td_error = td_target - self.Critic(state)
            td_error = td_error.cpu().flatten().numpy()
            A = [0]       # Advantage 优势估计

            '''done for GAE'''
            for dlt, done in zip(td_error[::-1], done.cpu().flatten().numpy()[::-1]):
                advantage = dlt + self.gamma * self.lambd * adv[-1] * (~done)
                A.append(advantage)
            A.reverse()
            A = copy.deepcopy(A[0:-1])
            A = torch.tensor(A).unsqueeze(1).float().to(self.dvc)
            td_target = A + value
            if self.adv_normalization:
                adv = (adv - adv.mean()) / ((adv.std() + 1e-4))  #sometimes helps 

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