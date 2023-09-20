from typing import Any, Optional, Tuple, List, Iterable

import torch
import torch.nn as nn
from torch import Tensor

class IPositionalEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x : Tuple[Tensor]): pass

class IQueryKeyEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x : Tuple[Tensor]): pass

