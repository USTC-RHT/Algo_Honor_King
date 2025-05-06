import numpy as np
from kaiwu_algo.model_based.given_model.dynamic_programming.dynamic_programming_algo import Algo
from kaiwu_algo.model_based.given_model.dynamic_programming.dynamic_programming_config import Config
from dataclasses import dataclass

@dataclass
class Reward:
    reward: float
    prob: float

@dataclass
class Trans:
    state: float
    prob: float
    
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
        self.trans_state_prob = [[[] for _ in range(self.action_size)] for _ in range(self.state_size)]     # Markov奖励过程状态转移概率分布 p(s'|s,a)
        self.trans_reward_prob = [[[] for _ in range(self.action_size)] for _ in range(self.state_size)]                # Markov奖励过程奖励转移概率分布 p(r|s,a)
        self.policy = np.ones((self.state_size,self.action_size)) / self.action_size
        self.value = np.zeros(self.state_size)

        for i in range(self.state_size):
            for j in range(self.action_size):
                if 25 <= i <= 34 and j == 2:
                    self.trans_reward_prob[i][j].append(Reward(-100,1))
                elif i == 36 and j == 1:
                    self.trans_reward_prob[i][j].append(Reward(-100,1))
                elif i == 47:
                    pass
                else:
                    self.trans_reward_prob[i][j].append(Reward(-1,1))
                if (25 <= i <= 34 and j == 2) or 37 <= i <= 47:
                    self.trans_state_prob[i][j] = []
                elif i == 36 and j == 1:
                    self.trans_state_prob[i][j] = []
                elif i == 36 and j == 2:
                    self.trans_state_prob[i][j].append(Trans(i,1))
                elif i % 12 == 0 and j == 3:
                    self.trans_state_prob[i][j].append(Trans(i,1))
                elif i < 12 and j == 0:
                    self.trans_state_prob[i][j].append(Trans(i,1))
                elif (i + 1) % 12 == 0 and j == 1:
                    self.trans_state_prob[i][j].append(Trans(i,1))
                else:
                    if j == 0:
                        self.trans_state_prob[i][j].append(Trans(i - 12,1))
                    elif j == 1:
                        self.trans_state_prob[i][j].append(Trans(i + 1,1))
                    elif j == 2:
                        self.trans_state_prob[i][j].append(Trans(i + 12,1))                
                    elif j == 3:
                        self.trans_state_prob[i][j].append(Trans(i - 1,1))
                

    def take_action(self,state):
        return np.argmax(self.policy[state])

    def update(self):
        algo = Algo([self.trans_state_prob,self.trans_reward_prob,self.policy,self.value],Config)
        algo.learn()

    def best_action(self,state):
        return np.argmax(self.policy[state])
    

    
