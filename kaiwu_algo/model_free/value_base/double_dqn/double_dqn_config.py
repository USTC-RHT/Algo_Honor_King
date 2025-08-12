class Config:
    learning_rate = 1e-2             # 学习率
    gamma = 0.98                     # 折扣因子
    epsilon = 0.01                   # epsilon-贪婪策略中的epsilon参数
    target_network_update_freq = 50  # 目标网络更新频率
    num_episodes = 200               # 智能体在环境中运行的序列的数量
    buffer_size = 5000               # 经验回放池的大小
    minimal_size = 1000              # 当经验回放池中的数据数量超过minimal_size后,才进行训练
    batch_size = 64                  # 训练需要抽取的经验批次大小
    ''' 在dqn_main中定义以下变量配置 '''
    state_dim = 0
    hidden_dim = 128
    action_dim = 0