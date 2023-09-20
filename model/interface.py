from typing import Any, Optional, Tuple, List, Iterable

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import abc

class IFeedForwardSubLayer():
    def forward(self, x:Tensor):
        raise NotImplementedError
    pass

class IAttentionSubLayer():
    @abc.abstractmethod
    def forward(self, xq : Tensor, xk : Tensor, xv : Tensor, recurrent_memory : Optional[Tensor] = None):
        raise NotImplementedError

