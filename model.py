import torch
import numpy as np
from torch import nn

class Model(nn.Module):
    def __init__(self,action_shape):
        super().__init__()