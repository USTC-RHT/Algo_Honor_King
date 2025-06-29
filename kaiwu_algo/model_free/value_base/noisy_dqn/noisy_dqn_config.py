class Config:
    tau = 0.005                      # 软更新系数
    learning_rate = 1e-4             # 学习率
    gamma = 0.99                     # 折扣因子
    epsilon = 0.01                   # epsilon-贪婪策略中的epsilon参数
    # target_network_update_freq = 10  # 目标网络更新频率
    update_every = 50                # 训练频率
    eval_interval = 1000             # 评估频率
    num_episodes = 8000               # 智能体在环境中运行的序列的数量
    buffer_size = int(6e5)           # 经验回放池的大小
    minimal_size = 3000              # 当经验回放池中的数据数量超过minimal_size后,才进行训练
    batch_size = 256                 # 训练需要抽取的经验批次大小
    hidden_layers = (200,200)        # 各个隐藏层的网络规模