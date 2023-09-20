from util.config import Factory

from typing import Any, Optional, Tuple, List, Iterable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class RMSNorm(nn.Module):
    def __init__(self, dim : int, weight_scaling : bool = True, eps = 1e-8):
        super().__init__()
        self.eps = eps
        self.dim = dim
        starting_scale = dim ** -0.5
        if weight_scaling:
            self.register_parameter("scale", nn.Parameter(torch.ones(dim) * starting_scale))
        else:
            self.scale = starting_scale

    def forward(self, x):
        assert(self.dim == x.size(-1))
        rms_norm = self.scale * x.norm(2, dim=-1, keepdim=True)
        return x / rms_norm.clamp(self.eps)
    
    @staticmethod
    def F(x, eps = 1e-8):
        rms_norm = (x.size(-1) ** -0.5) * x.norm(2, dim=-1, keepdim=True)
        return x / rms_norm.clamp(eps)

class Norm(nn.Module):
    def __init__(self, dim : int, weight_scaling : bool = True, eps = 1e-8):
        super().__init__()
        self.eps = eps
        if weight_scaling:
            self.register_parameter("scale", nn.Parameter(torch.ones(dim)))
        else:
            self.scale = 1

    def forward(self, x):
        return self.scale * x / x.norm(2, dim=-1, keepdim=True).clamp(self.eps)

    @staticmethod
    def F(x):
        # assumes that vector 'normally' has length 1, not length vec.size(-1)**0.5 (which would be if every component had an average absolute value of 1!)
        return x / x.norm(2, dim=-1, keepdim=True).clamp(1e-8)
