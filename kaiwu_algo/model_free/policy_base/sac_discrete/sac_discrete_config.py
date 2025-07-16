class Config:
    gamma = 0.99                           # 折扣因子
    tau = 0.005                            # 目标网络软更新系数
    alpha = 0.2                            # 正则化系数，用来控制熵的重要程度
    adaptive_alpha = True                  # 是否使用自适应 alpha系数
    target_entropy = 0                     # 策略熵的目标值(在sac_continuous_agent中改写)
    buffer_size = int(1e6)                 # 经验回放池的大小
    actor_hidden_layers = (200,200)        # actor各个隐藏层的网络规模
    critic_hidden_layers = (200,200)       # critic各个隐藏层的网络规模
    actor_learning_rate = 3e-4             # actor模型学习率
    critic_learning_rate = 3e-4            # critic模型学习率
    log_actor_learning_rate = 3e-4         # 自适应 alpha学习率
    random_steps = 1e4                     # 开始训练之前的随机探索步数
    Max_train_steps = 4e5                  # 最大训练步数
    batch_size = 256                       # 批次大小
    update_every = 50                      # 训练频率
    eval_interval = 2000                   # 评估频率
    LOG_SIGMA_RANGE = (-20,2)              # 限制 log_sigma的范围，防止数值不稳定
    ''' 开始训练之前的随机探索步数
    minimal_size = 10 * env.spec.max_episode_steps '''