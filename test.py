
import torch
from torch.distributions import Categorical

# 假设 batch_size = 3，action_dim = 4
probs = torch.tensor([
    [1, 0, 0, 0],   # 第一个样本的动作分布
    [0.25, 0.25, 0.25, 0.25],  # 第二个样本（最大熵）
    [0.9, 0.05, 0.03, 0.02],   # 第三个样本（最小熵）
])

dist = Categorical(probs=probs)
entropy = dist.entropy()

print("entropy:", entropy)
print("entropy shape:", entropy.shape)
