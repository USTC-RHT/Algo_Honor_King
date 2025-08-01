class Config:
    gamma = 0.99                           # 折扣因子
    lambd = 0.95                           # GAE平滑系数
    actor_hidden_layers = (64,64)          # actor各个隐藏层的网络规模
    critic_hidden_layers = (64,64)         # critic各个隐藏层的网络规模
    actor_learning_rate = 5e-4             # actor模型学习率
    critic_learning_rate = 5e-4            # critic模型学习率
    Max_train_steps = int(1e6)             # 最大训练步数
    num_epochs = 15                        # PPO更新的轮数(epoch)
    batch_size = 32                        # replay_buffer储存的batch大小
    mini_batch_size = 8                    # mini_batch update的尺寸
    clip_rate = 0.2                        # PPO-截断 PPO-Clip rate
    entropy_coef = 0.01                    # actor中 entropy loss的系数(熵系数)
    clip_grad_max_norm = 10                # 梯度裁剪的最大梯度上限
    Advantage_Normal = False               # 是否进行优势的归一化
    eval_interval = 1000                   # 评估频率
    use_agent_specific = True              # 是否在Critic模型输入中增加智能体特有的观测
    use_lr_decay = True                    # 是否对学习率进行衰减
    use_reward_norm = True                 # 是否对奖励进行归一化
    use_adam_eps = True                    # 是否在 Adam 优化器中设置 eps
    adam_eps = 1e-5                        # Adam 优化器中设置的 eps大小
    ''' 以下参数均在mappo_main.py中定义 '''
    agent_num = 0                          # 智能体的数量
    state_dim = 0                          # 全局状态维度
    obs_dim_n = []                         # 各个智能体 obs的维度列表
    action_dim_n = []                      # 各个智能体 action的维度列表
    episode_limit = 0                      # 每个episode的最大步数
