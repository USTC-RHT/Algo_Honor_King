class Config:
    gamma = 0.95                           # 折扣因子
    noise = 0.1                            # 探索噪声
    tau = 0.01                             # 目标网络软更新系数
    buffer_size = int(1e5)                 # 经验回放池的大小
    minimal_size = int(4e3)                # 开始训练之前的随机探索步数
    actor_hidden_layers = (64,64)          # actor各个隐藏层的网络规模
    critic_hidden_layers = (64,64)         # critic各个隐藏层的网络规模
    actor_learning_rate = 1e-2             # actor模型学习率
    critic_learning_rate = 1e-2            # critic模型学习率
    Max_train_steps = 5e6                  # 最大训练步数
    batch_size = 1024                      # 批次大小
    update_every = 100                     # 训练频率
    eval_interval = 1000                   # 评估频率

