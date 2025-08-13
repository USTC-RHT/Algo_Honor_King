class Config:
    learning_rate = 1e-4             # 学习率
    gamma = 0.99                     # 折扣因子
    epsilon = 0.01                   # epsilon-贪婪策略中的epsilon参数
    target_network_update_freq = 50  # 目标网络更新频率
    num_episodes = 8000              # 智能体在环境中运行的序列的数量
    buffer_size = 100000
    minimal_size = 3000
    batch_size = 256
    alpha = 0.6                      # alpha for PER
    beta = 0.4                       # beta for PER 
    prioritized_replay_eps = 1e-6
    eval_interval = 1000             # 评估频率
    ''' 在dqn_main中定义以下变量配置 '''
    state_dim = 0
    hidden_dim = 256
    action_dim = 0