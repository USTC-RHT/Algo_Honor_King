import numpy as np
from q_learning_algo import Algo
from q_learning_config import Config
from dataclasses import dataclass

@dataclass
class Data:
    state: int
    action: int
    reward: float
    next_state: int
    done: bool
    
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
        self.Q_table = np.ones([self.state_size, self.action_size])

    def take_action(self,state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_size)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def update(self,obs,action,r,next_obs,done):
        # 创建一个 Data 实例
        list_sample_data = [Data(state=obs, action=action, reward=r, next_state=next_obs, done=done)]
        algo = Algo(self.Q_table,Config)
        algo.learn(list_sample_data)
    
    def best_action(self,state):
        return np.argmax(self.Q_table[state])
    

    
