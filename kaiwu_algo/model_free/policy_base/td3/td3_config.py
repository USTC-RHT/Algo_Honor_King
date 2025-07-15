class Config:
    gamma = 0.99                           # 折扣因子
    max_action = 0                         # 最大动作值(在td3_agent中改写)
    explore_noise = 0.15                   # 探索噪声
    explore_noise_decay = 0.998            # 探索噪声的衰减率
    target_policy_noise = 0                # 目标策略的噪声(在td3_agent中改写)
    target_policy_noise_clip = 0           # 目标策略的噪声裁剪(在td3_agent中改写)
    tau = 0.005                            # 目标网络软更新系数
    buffer_size = int(1e6)                 # 经验回放池的大小
    actor_hidden_layers = (256,256)        # actor各个隐藏层的网络规模
    critic_hidden_layers = (256,256)       # critic各个隐藏层的网络规模
    actor_learning_rate = 1e-4             # actor模型学习率
    critic_learning_rate = 1e-4            # critic模型学习率
    Max_train_steps = 5e6                  # 最大训练步数
    batch_size = 256                       # 批次大小
    update_every = 50                      # 训练频率
    eval_interval = 2000                   # 评估频率
    delay_freq = 2                         # (延迟)更新 Actor and Target Net的频率
    ''' 开始训练之前的随机探索步数
    minimal_size = 10 * env.spec.max_episode_steps '''

