class Config:
    gamma = 0.95                           # 折扣因子
    tau = 0.01                             # 目标网络软更新系数
    episode_limit = 25                     # 每个episode的最大步数限制
    buffer_size = int(1e6)                 # 经验回放池的大小
    minimal_size = 1024                    # 开始训练之前的随机探索步数
    actor_hidden_layers = (64,64)          # actor各个隐藏层的网络规模
    critic_hidden_layers = (64,64)         # critic各个隐藏层的网络规模
    actor_learning_rate = 5e-4             # actor模型学习率
    critic_learning_rate = 5e-4            # critic模型学习率
    Max_train_steps = 1e6                  # 最大训练步数
    batch_size = 1024                      # 批次大小
    update_every = 1                       # 训练频率
    use_orthogonal_init = True             # 神经网络是否使用正交初始化
    eval_interval = 1000                   # 评估频率
    noise = 0.2                            # 探索噪声
    noise_init = 0.2                       # 探索噪声的初始值
    noise_min = 0.05                       # 探索噪声的最小值
    noise_decay_steps = int(3e5)           # 探索噪声衰减到最小值所需的步数
    use_noise_decay = True                 # 是否对探索噪声进行衰减
    use_grad_clip = True                   # 是否进行梯度裁剪
    clip_grad_max_norm = 10                # 梯度裁剪的最大梯度上限
    ''' 以下参数均在maddpg_main.py中定义 '''
    obs_dim_n = []                         # 各个智能体obs的维度列表
    action_dim_n = []                      # 各个智能体action的维度列表
    agent_num = 0                          # 智能体的数量
    max_action = 1                         # 智能体动作的最大值