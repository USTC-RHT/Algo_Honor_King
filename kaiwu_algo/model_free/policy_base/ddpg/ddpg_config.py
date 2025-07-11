class Config:
    gamma = 0.99                           # 折扣因子
    noise = 0.1                            # 探索噪声
    buffer_size = int(5e5)                 # 经验回放池的大小
    minimal_size = int(5e4)                # 开始训练之前的随机探索步数
    actor_hidden_layers = (400,300)        # actor各个隐藏层的网络规模
    critic_hidden_layers = (400,300)       # critic各个隐藏层的网络规模
    actor_learning_rate = 1e-3             # actor模型学习率
    critic_learning_rate = 1e-3            # critic模型学习率
    Max_train_steps = 5e6                  # 最大训练步数
    batch_size = 128                       # sliced trajectory的长度/批次大小
    eval_interval = 1000                   # 评估频率

