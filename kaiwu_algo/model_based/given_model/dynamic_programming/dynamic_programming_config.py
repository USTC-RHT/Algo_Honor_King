class Config:
    # algo_name = "value_iteration"     # 值迭代算法
    algo_name = "policy_iteration"  # 策略迭代算法
    state_size = 48                   # 状态空间大小
    action_size = 4                   # 动作空间大小
    gamma = 0.9                       # 折扣因子
    theta = 1e-3                      # 收敛阈值
    max_iter_num = 10000              # 最大迭代次数