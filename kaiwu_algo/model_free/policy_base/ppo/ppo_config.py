class Config:
    gamma = 0.99                           # 折扣因子
    lambd = 0.95                           # GAE平滑系数
    L2_reg = 0                             # critic中 L2正则化系数
    num_episodes = 1000                    # 智能体在环境中运行的序列的数量
    actor_hidden_layers = (64,64)          # actor各个隐藏层的网络规模
    critic_hidden_layers = (64,64)         # critic各个隐藏层的网络规模
    Max_train_steps = 5e7                  # 最大训练步数
    num_epochs = 10                        # PPO更新的轮数(epoch)
    batch_size = 64                        # sliced trajectory的长度/批次大小
    clip_rate = 0.2                        # PPO-截断 PPO-Clip rate
    entropy_coef = 0                       # actor中 entropy loss的系数
    clip_grad_max_norm = 40                # 梯度裁剪的最大梯度上限