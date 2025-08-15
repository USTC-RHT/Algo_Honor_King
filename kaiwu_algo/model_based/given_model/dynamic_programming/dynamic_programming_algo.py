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

        ''' 下面两项均为二维矩阵 两个维度分别对应状态和动作 '''
        self.policy = model[2]                           # 策略表
        self.value = model[3]                            # 价值表
        

    def learn(self):
        # 确认算法名称在支持的范围内
        assert self.algo_name in ["value_iteration", "policy_iteration"], "Invalid algorithm"

        # 根据指定算法名称调用不同算法
        if self.algo_name == "value_iteration":
            self.value_iteration()
        elif self.algo_name == "policy_iteration":
            self.policy_iteration()
    
    def value_iteration(self):
        # 记录当前迭代次数
        iteration_num = 0
        while iteration_num < self.max_iter_num:  # 最大迭代次数限制
            delta = 0  # 追踪单次迭代最大的状态值变化

            # 遍历所有状态
            for state in range(self.state_size):
                value_pre = np.copy(self.value[state])  # 保存当前状态的旧value
                action_value_list = []
                # 计算当前状态下每个动作的动作价值
                for action in range(self.action_size):
                    action_value_list.append(self.get_action_value(state, action))
                
                action_value_list = np.array(action_value_list)
                # 【策略改进】选择使state价值最大的动作，并将策略概率置为1
                argmax_index = int(np.argmax(action_value_list))
                self.policy[state] = np.zeros(self.action_size)
                self.policy[state][argmax_index] = 1
                # 【价值更新】用该动作的价值更新当前状态的价值
                self.value[state] = action_value_list[argmax_index]

                # 记录本轮最大价值更新幅度
                delta = max(delta, abs(value_pre - self.value[state]))

            # 当最大变化量小于阈值，收敛，跳出循环
            if delta < self.theta:
                break

            # 每10次输出一次进度
            if iteration_num % 10 == 0:
                print("Iteration {}".format(iteration_num))
            iteration_num += 1
        return 

    def policy_iteration(self):
        iteration_num = 0
        while iteration_num < self.max_iter_num:
            pre_policy = np.copy(self.policy)    # 备份当前策略
            self.policy_evaluation(pre_policy)   # 用当前策略评估得到新的value
            new_policy = self.policy_improvement()  # 基于新的value改进策略

            self.policy[:] = new_policy[:]    # 更新策略

            # 如果策略前后没有明显变化则收敛，退出
            if np.allclose(pre_policy, new_policy, atol=1e-4):
                break

            if iteration_num % 10 == 0:
                print("Iteration {}".format(iteration_num))
            iteration_num += 1
        return 

    def policy_evaluation(self, policy):
        # 初始化状态价值表为0
        self.value = np.zeros(self.state_size)
        delta = self.theta + 1    # 用于判断是否收敛

        # 当所有状态value的变化幅度都小于theta时终止评估
        while delta > self.theta:
            delta = 0
            # 遍历所有状态
            for state in range(self.state_size):
                v = 0
                # 按照当前策略的动作概率求加权期望价值
                for action in range(self.action_size):
                    v += policy[state][action] * self.get_action_value(state, action)

                # 记录最大价值变化幅度
                delta = max(delta, abs(v - self.value[state]))

                # 更新该状态的value
                self.value[state] = v
        return

    def policy_improvement(self):
        # 初始化新策略为均匀分布
        policy = np.ones((self.state_size, self.action_size)) / self.action_size
        for state in range(self.state_size):
            action_value_list = []
            # 为当前状态计算所有动作的value
            for action in range(self.action_size):
                action_value_list.append(self.get_action_value(state, action))
            
            action_value_list = np.array(action_value_list)
            # 只保留最大动作的概率为1，其它为0
            argmax_index = int(np.argmax(action_value_list))
            policy[state] = np.zeros(self.action_size)
            policy[state][argmax_index] = 1
        return policy

    def get_action_value(self, state, action):
        action_value = 0
        # 计算当前“状态-动作”下所有可能获得的奖励的期望
        for item in self.trans_reward_prob[state][action]:
            action_value += item.reward * item.prob
            # 如果没有后继状态，直接返回
        if len(self.trans_state_prob[state][action]) == 0:
            return action_value
        else:
            # 有后继状态，则加上折扣后的未来价值
            for item in self.trans_state_prob[state][action]:
                action_value += self.gamma * item.prob * self.value[item.state]
            return action_value
            
            
        
            