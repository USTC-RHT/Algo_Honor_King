import numpy as np
from sarsa_algo import Algo
from sarsa_config import Config
from dataclasses import dataclass

@dataclass
class Data:
    state: int
    action: int
    reward: float
    next_state: int
    next_action: int
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
        self.algo = Algo(self.Q_table,Config)

    def predict(self,state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_size)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def learn(self,obs,action,r,next_obs,next_action,done):
        # 创建一个 Data 实例
        list_sample_data = [Data(state=obs, action=action, reward=r, next_state=next_obs, next_action=next_action, done=done)]
        self.algo.learn(list_sample_data)
        return
    
    def exploit(self,state):
        return np.argmax(self.Q_table[state])
    

    
