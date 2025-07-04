class Config:
    gamma = 0.99                           # 折扣因子
    lambd = 0.95                           # GAE平滑系数
    num_episodes = 1000                    # 智能体在环境中运行的序列的数量
    actor_hidden_layers = (64,64)          # actor各个隐藏层的网络规模
    critic_hidden_layers = (64,64)         # critic各个隐藏层的网络规模
    Max_train_steps = 5e7                  # 最大训练步数