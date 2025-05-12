class Config:
    gamma = 0.98                     # 折扣因子
    num_episodes = 1000            #智能体在环境中运行的序列的数量

    '''
    class Config:
    """
    代码包下的所有配置
    """
    # Discount factor GAMMA in RL
    # RL中的回报折扣GAMMA
    GAMMA = 0.95

    # Initial learning rate
    # 初始的学习率
    START_LR = 5e-4

    VALUE_LOSS_COEFF = 0.5
    ENTROPY_LOSS_COEFF = 0.025
    '''