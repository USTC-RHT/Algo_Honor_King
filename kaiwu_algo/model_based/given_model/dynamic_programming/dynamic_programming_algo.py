from base_algo import BaseAlgo
import numpy as np

class Algo(BaseAlgo):
    def __init__(self, model, config, optimizer=None, device=None, logger=None, monitor=None): 
        # 离散表格型强化学习算法 不需要神经网络与优化器
        self.config = config                             # 配置文件
        self.algo_name = config.algo_name                # 算法名字（值迭代算法或策略迭代算法）
        self.gamma = config.gamma                        # 折扣因子
        self.theta = config.theta                        # 收敛阈值
        self.state_size = config.state_size              # 状态空间大小
        self.action_size = config.action_size            # 动作空间大小
        self.max_iter_num = config.max_iter_num          # 最大迭代次数
        """
        下面两项均为二维矩阵 两个维度分别对应状态和动作 
        每个矩阵元素是一个列表 列表每个元素含有属性 reward/state + prob 
        """
        self.trans_state_prob = model[0]                 # Markov奖励过程状态转移概率分布 p(s'|s,a)
        self.trans_reward_prob = model[1]                # Markov奖励过程奖励转移概率分布 p(r|s,a)

        # 下面两项均为二维矩阵 两个维度分别对应状态和动作
        self.policy = model[2]                           # 策略表
        self.value = model[3]                            # 价值表
        

    def learn(self):
        assert self.algo_name in ["value_iteration", "policy_iteration"], "Invalid algorithm"

        if self.algo_name == "value_iteration":
            self.value_iteration()
        elif self.algo_name == "policy_iteration":
            self.policy_iteration()
    
    def value_iteration(self):
        iteration_num = 0
        while iteration_num < self.max_iter_num:
            delta = 0

            for state in range(self.state_size):
                value_pre = np.copy(self.value[state])
                action_value_list = []
                for action in range(self.action_size):
                    action_value_list.append(self.get_action_value(state, action))
                
                action_value_list = np.array(action_value_list)
                # policy update
                argmax_index = int(np.argmax(action_value_list))
                self.policy[state] = np.zeros(self.action_size)
                self.policy[state][argmax_index] = 1
                # value update
                self.value[state] = action_value_list[argmax_index]

                delta = max(delta, abs(value_pre - self.value[state]))

            if delta < self.theta:
                break

            if iteration_num % 10 == 0:
                print("Iteration {}".format(iteration_num))
            iteration_num += 1
        return 



    def policy_iteration(self):
        iteration_num = 0
        while iteration_num < self.max_iter_num:
            pre_policy = np.copy(self.policy)
            self.policy_evaluation(pre_policy)
            new_policy = self.policy_improvement()

            self.policy[:] = new_policy[:]

            if np.allclose(pre_policy, new_policy, atol=1e-4):
                break

            if iteration_num % 10 == 0:
                print("Iteration {}".format(iteration_num))
            iteration_num += 1
        return 


    def policy_evaluation(self,policy):
        # 初始化状态价值表
        self.value = np.zeros(self.state_size)
        delta = self.theta + 1

        while delta > self.theta:
            delta = 0
            # 遍历所有状态
            for state in range(self.state_size):
                v = 0
                # 遍历给定状态的所有动作
                for action in range(self.action_size):
                    v += policy[state][action] * self.get_action_value(state, action)

                # 计算给定状态的旧值和新值之间的差值
                delta = max(delta, abs(v - self.value[state]))

                # 更新状态价值表
                self.value[state] = v
        return

    def policy_improvement(self):
        policy = np.ones((self.state_size,self.action_size)) / self.action_size
        for state in range(self.state_size):
            action_value_list = []
            for action in range(self.action_size):
                action_value_list.append(self.get_action_value(state, action))
        
            action_value_list = np.array(action_value_list)
            # policy update
            argmax_index = int(np.argmax(action_value_list))
            policy[state] = np.zeros(self.action_size)
            policy[state][argmax_index] = 1
        return policy



    def get_action_value(self,state, action):
        action_value = 0
        for item in self.trans_reward_prob[state][action]:
            action_value += item.reward * item.prob
        if len(self.trans_state_prob[state][action]) == 0:
            return action_value
        else:
            for item in self.trans_state_prob[state][action]:
                action_value += self.gamma * item.prob * self.value[item.state]
            return action_value
        
        
    
        