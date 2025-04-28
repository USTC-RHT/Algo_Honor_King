from base_algo import BaseAlgo

class Algo(BaseAlgo):
    def __init__(self, model, config, optimizer=None, device=None, logger=None, monitor=None): 
        # 离散表格型强化学习算法 不需要神经网络与优化器
        self.config = config                             # 配置文件
        self.gamma = config.gamma                        # 折扣因子
        self.Q = model[0]                                # 在这里模型 model包含 Q值表与访问次数(visit)表，在智能体中定义
        self.visit = model[1]     


    def learn(self, list_sample_data):
        """
        使用蒙特卡洛控制 更新Q值表方式如下:
        1.采用首次访问(first visit)策略
        2.增量式更新
        Q值表更新的准确性依赖于是否能大量采样到所有状态-动作对,由epsilon-贪婪策略保证
        - list_sample_data是一个episode列表:[(s_0,a_0,r_1),...,(s_{T-1},a_{T-1},r_T)]
        使用以下公式计算返回值：
        - Return = r_{t+1} + gamma * r_{t+2} + ... + gamma^(T-t-1) * r_T
        """
        self.sample_data_check(list_sample_data)    # 检查样本字段是否符合要求
        Return, state_action_return = 0, []        # state_action_return是一个储存(state,action,return)的列表

        # 计算每个状态-动作对的返回值  倒序累加利于计算
        for sample in reversed(list_sample_data):
            Return = self.gamma * Return + sample.reward
            state_action_return.append((sample.state, sample.action, Return))

        state_action_return.reverse()

        # 更新Q值表
        seen_state_action = set()
        for state, action, Return in state_action_return:
            if (state, action) not in seen_state_action:
                # 每个状态-动作对访问次数更新
                self.visit[state][action] += 1

                # 增量式更新 递增均值
                self.Q[state, action] = self.Q[state, action] + (Return - self.Q[state, action]) / self.visit[state, action]
                seen_state_action.add((state, action))
        
    
    def sample_data_check(self,list_sample_data):
        if not isinstance(list_sample_data, list):
            raise TypeError("Expected data to be a list")  # 数据类型错误 数据需要是一个列表！
        if len(list_sample_data) == 0:
            raise ValueError("Data cannot be empty")  # 数据为空错误
        