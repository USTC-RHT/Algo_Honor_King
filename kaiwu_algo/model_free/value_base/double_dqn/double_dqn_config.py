class Config:
    learning_rate = 2e-3             # 学习率
    gamma = 0.98                     # 折扣因子
    epsilon = 0.01                   # epsilon-贪婪策略中的epsilon参数
    target_network_update_freq = 10  # 目标网络更新频率
    num_episodes = 500               # 智能体在环境中运行的序列的数量
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64