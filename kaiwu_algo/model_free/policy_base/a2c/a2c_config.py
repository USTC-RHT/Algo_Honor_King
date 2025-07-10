class Config:
    gamma = 0.98                           # 折扣因子
    L2_reg = 0                             # critic中 L2正则化系数
    actor_hidden_layers = (128)            # actor各个隐藏层的网络规模
    critic_hidden_layers = (128)           # critic各个隐藏层的网络规模
    actor_learning_rate = 1e-3             # actor模型学习率
    critic_learning_rate = 1e-2            # critic模型学习率
    num_episodes = 1000                    # 智能体在环境中运行的序列的数量
    entropy_coef = 0                       # actor中 entropy loss的系数(熵系数)
    entropy_coef_decay = 0.99              # 熵系数的衰减率
    clip_grad_max_norm = 40                # 梯度裁剪的最大梯度上限
    Advantage_Normal = False               # 是否进行优势的归一化