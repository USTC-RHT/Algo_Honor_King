class Config:
    gamma = 0.99                           # 折扣因子
    tau = 0.005                            # 目标网络软更新系数
    epsilon = 0.0                          # epsilon-贪婪策略中的epsilon参数
    epsilon_decay_steps = 50000            # epsilon衰减到最小值所需步数
    epsilon_min = 0.05                     # epsilon的最小值
    hidden_layers = (64,64)                # 各个隐藏层的网络规模
    learning_rate = 5e-4                   # 模型学习率
    Max_train_steps = int(1e6)             # 最大训练步数
    batch_size = 32                        # replay_buffer储存的batch大小(这里代表有多少个episode)
    clip_grad_max_norm = 10                # 梯度裁剪的最大梯度上限
    eval_interval = 1000                   # 评估频率
    use_rnn = True                         # 神经网络是否使用rnn模块推理训练
    use_agent_specific = True              # 是否在Critic模型输入中增加智能体特有的观测
    use_lr_decay = False                   # 是否对学习率进行衰减
    use_reward_norm = True                 # 是否对奖励进行归一化
    add_last_action = True                 # 是否在输入中加入上次的动作
    add_agent_id = True                    # 是否在输入中加入agent_id
    target_network_update_freq = 10        # 目标网络更新频率
    ''' 以下参数均在mappo_main.py中定义 '''
    agent_num = 0                          # 智能体的数量
    state_dim = 0                          # 全局状态维度
    obs_dim_n = []                         # 各个智能体 obs的维度列表
    action_dim_n = []                      # 各个智能体 action的维度列表
    episode_limit = 0                      # 每个episode的最大步数
