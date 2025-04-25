from base_algo import BaseAlgo
import numpy as np

class Algo(BaseAlgo):
    def __init__(self, model, config, optimizer=None, device=None, logger=None, monitor=None): 
        # 离散表格型强化学习算法 不需要神经网络与优化器
        self.train_step = 0

        self.config = config                             # 配置文件
        self.gamma = config.gamma                        # 折扣因子
        self.learning_rate = config.learning_rate        # 学习率
        self.epsilon = config.epsilon                    # epsilon-贪婪策略中的epsilon参数
        self.Q = model                                   # 在这里模型 model代表 Q值表，在智能体中定义


    def learn(self, list_sample_data):
        """
        使用给定的数据更新Q表格:
        list_sample_data -> 每个样本的格式均是(state, action, reward, next_state, done)
        使用以下公式更新Q值:
        Q(s,a) := Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        其中：
        Q(s,a) 表示状态s下采取动作a的Q值
        lr 是学习率(learning rate), 用于控制每次更新的幅度
        R(s,a) 是在状态s下采取动作a所获得的奖励
        gamma 是折扣因子, 用于平衡当前奖励和未来奖励的重要性
        max Q(s',a') 表示在新状态s'下采取所有可能动作a'的最大Q值
        done 表示episode是否结束
        """
        self.sample_data_check(list_sample_data)    # 检查样本字段是否符合要求
        sample = list_sample_data[0]
        state, action, reward = sample.state, sample.action, sample.reward 
        next_state, done = sample.next_state, sample.done

        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.Q[next_state, :])
        
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.learning_rate * td_error
        return
        
    def sample_data_check(self,list_sample_data):
        if not isinstance(list_sample_data, list):
            raise TypeError("Expected data to be a list")  # 数据类型错误 数据需要是一个列表！
        if len(list_sample_data) == 0:
            raise ValueError("Data cannot be empty")  # 数据为空错误
        