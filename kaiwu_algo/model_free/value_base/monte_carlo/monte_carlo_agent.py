import numpy as np
from monte_carlo_algo import Algo
from monte_carlo_config import Config
from dataclasses import dataclass

@dataclass
class Data:
    state: int
    action: int
    reward: float
    
class Agent:
    def __init__(self,env_name):
        # 选择需要运行的环境
        env_name = "CliffWalking-v0"    # "FrozenLake-v1"
        if env_name == "CliffWalking-v0":
            state_size = 48            # 状态维度
            action_size = 4            # 动作维度
        if env_name == "FrozenLake-v1":
            state_size = 16
            action_size = 4
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = Config.epsilon
        self.Q_table = np.zeros([self.state_size, self.action_size])
        self.visit = np.zeros_like(self.Q_table)

    def take_action(self,state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_size)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def update(self,Episode):
        # 创建一个 Data 实例
        list_sample_data = [Data(state=obs, action=action, reward=r) for (obs,action,r) in Episode]
        algo = Algo([self.Q_table,self.visit],Config)
        algo.learn(list_sample_data)
    
    def best_action(self,state):
        return np.argmax(self.Q_table[state])
    

    
