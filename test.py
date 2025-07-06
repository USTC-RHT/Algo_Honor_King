import torch

# 构造小型张量
reward = torch.tensor([1.0, 2.0, 3.0])           # shape: [3]
next_value = torch.tensor([[10.0], [20.0], [30.0]])  # shape: [3, 1]
dw = torch.tensor([0, 1, 0])               # shape: [3]
gamma = 0.99

temp = gamma * next_value * (1 - dw)
# 执行表达式
TD_target = reward + temp

# 打印各部分形状和结果
print("reward.shape:", reward.shape)
print("next_value.shape:", next_value.shape)
print("dw.shape:", dw.shape)
print("TD_target:", TD_target)
print("TD_target.shape:", TD_target.shape)