from base_algo import BaseAlgo
import torch
import copy
import math
import numpy as np
from torch.distributions import Categorical

class Algo(BaseAlgo):
    def __init__(self, model, config, optimizer=None, device=None, logger=None, monitor=None): 
        self.config = config                             # 配置文件
        self.gamma = config.gamma                        # 折扣因子
        self.lambd = config.lambd                        # GAE平滑系数
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.clip_rate = config.clip_rate
        self.entropy_coef = config.entropy_coef
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
        actor_loss = self.calculate_loss(list_sample_data, model_output_data)	    # 计算loss
        self.actor_optimizer.zero_grad()					                        # 清空梯度
        actor_loss.mean().backward()												
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)                 # ????
        self.actor_optimizer.step()						                            # 更新模型
        self.train_step += 1						                                # 更新计数器
    
    def calculate_loss(self,list_sample_data, model_output_data):
        Return, loss = 0, 0
        i = -1       

        states, actions, rewards, next_states, logprob_actions, dws, dones =\
            zip(*[(sample_data.state, sample_data.action, sample_data.reward, 
                sample_data.next_state,sample_data.logprob_a, sample_data.dw,
                int(sample_data.done)) for sample_data in list_sample_data])

        state = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)         
        action = torch.tensor(np.array(actions), dtype=torch.int, device=self.device)     
        reward = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)       
        next_state = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        logprob_a = torch.tensor(np.array(logprob_actions), dtype=torch.float32, device=self.device)
        dw = torch.tensor(np.array(dws), dtype=torch.int, device=self.device)  
        dones = np.array(dones)

        ''' Use TD + GAE + LongTrajectory to compute Advantage and TD target'''
        with torch.no_grad():
            value = self.Critic(state)
            next_value = self.Critic(next_state)
            '''dw(dead and win) for TD_target and Adv'''
            TD_target = reward + self.gamma * next_value * (1 - dw)
            TD_error = TD_target - value
            TD_error = TD_error.cpu().flatten().numpy()

            '''Advantage 优势估计'''
            A = [0]       
            '''done for GAE'''
            for td_error, done in zip(TD_error[::-1], dones[::-1]):
                advantage = td_error + self.gamma * self.lambd * A[-1] * (1 - done)
                A.append(advantage)
            A.reverse()
            '''去除初始数组中的0元素'''
            A = copy.deepcopy(A[0:-1])    
            A = torch.tensor(A).unsqueeze(1).float().to(self.device)
            TD_target = A + value
            # if self.Advantage_Normal:
            #     A= (A - A.mean()) / ((A.std() + 1e-4))  #sometimes helps 

        '''PPO update'''
        '''Slice long trajectopy into short trajectory and perform mini-batch PPO update'''
        traj_len = dones.shape[0]
        optim_iter_num = int(math.ceil(traj_len / self.batch_size))

        for _ in range(self.num_epochs):
            '''Shuffle the trajectory, Good for training
            perm : short for permutation'''
            perm = np.arange(traj_len)
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(self.device)
            state, action, TD_target, A, old_prob = \
                state[perm].clone(), action[perm].clone(), \
                TD_target[perm].clone(), A[perm].clone(), logprob_a[perm].clone()

            '''mini-batch PPO update'''
            for i in range(optim_iter_num):
                index = slice(i * self.batch_size, min((i + 1) * self.batch_size, traj_len))

                '''actor loss'''
                new_prob = self.Actor(state[index])
                new_prob_a = new_prob.gather(1, action[index])
                ratio = torch.exp(torch.log(new_prob_a) - torch.log(old_prob[index]))  # a/b == exp(log(a)-log(b))

                surr1 = ratio * A[index]
                '''PPO-截断(PPO-Clip)'''
                surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * A[index]
                ppo_loss = -torch.min(surr1, surr2)

                entropy = Categorical(probs=new_prob).entropy()
                entropy_loss = - self.entropy_coef * entropy
                actor_loss = ppo_loss + entropy_loss

                '''critic loss'''
                c_loss = (self.Critic(state[index]) - TD_target[index]).pow(2).mean()
                for name, param in self.Critic.named_parameters():
                    if 'weight' in name:
                        c_loss += param.pow(2).sum() * self.l2_reg

                self.critic_optimizer.zero_grad()
                c_loss.backward()
                self.critic_optimizer.step()

        return actor_loss
    
    def sample_data_check(self,list_sample_data):
        if not isinstance(list_sample_data, list):
            raise TypeError("Expected data to be a list")  # 数据类型错误 数据需要是一个列表！
        if len(list_sample_data) == 0:
            raise ValueError("Data cannot be empty")  # 数据为空错误