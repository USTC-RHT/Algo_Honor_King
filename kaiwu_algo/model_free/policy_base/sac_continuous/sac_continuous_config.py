class Config:
    gamma = 0.99                           # 折扣因子
    tau = 0.005                            # 目标网络软更新系数
    buffer_size = int(1e6)                 # 经验回放池的大小
    actor_hidden_layers = (256,256)        # actor各个隐藏层的网络规模
    critic_hidden_layers = (256,256)       # critic各个隐藏层的网络规模
    actor_learning_rate = 3e-4             # actor模型学习率
    critic_learning_rate = 3e-4            # critic模型学习率
    Max_train_steps = 5e6                  # 最大训练步数
    batch_size = 256                       # 批次大小
    update_every = 50                      # 训练频率
    eval_interval = 2500                   # 评估频率
    LOG_SIGMA_RANGE = (-20,2)              # 限制 log_sigma的范围，防止数值不稳定
    ''' 开始训练之前的随机探索步数
    minimal_size = 10 * env.spec.max_episode_steps '''

