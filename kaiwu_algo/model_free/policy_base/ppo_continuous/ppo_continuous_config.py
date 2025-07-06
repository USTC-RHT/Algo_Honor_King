class Config:
    gamma = 0.99                           # 折扣因子
    lambd = 0.95                           # GAE平滑系数
    L2_reg = 0                             # critic中 L2正则化系数
    actor_hidden_layers = (64,64)          # actor各个隐藏层的网络规模
    critic_hidden_layers = (64,64)         # critic各个隐藏层的网络规模
    actor_learning_rate = 1e-4             # actor模型学习率
    critic_learning_rate = 1e-4            # critic模型学习率
    Max_train_steps = 5e7                  # 最大训练步数
    num_epochs = 10                        # PPO更新的轮数(epoch)
    batch_size = 64                        # sliced trajectory的长度/批次大小
    clip_rate = 0.2                        # PPO-截断 PPO-Clip rate
    entropy_coef = 0                       # actor中 entropy loss的系数(熵系数)
    entropy_coef_decay = 0.99              # 熵系数的衰减率
    clip_grad_max_norm = 40                # 梯度裁剪的最大梯度上限
    Advantage_Normal = False               # 是否进行优势的归一化
    eval_interval = 1000                   # 评估频率
    max_traj_len = 2048                    # 长轨迹的最大长度
