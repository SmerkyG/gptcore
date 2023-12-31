from typing import Any, Optional, Tuple, List, Iterable

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import abc

class IModule():
    @abc.abstractmethod
    def forward(self, x : Any):
        raise NotImplementedError
    
    @abc.abstractmethod
    def __call__(self, x : Any):
        raise NotImplementedError

class IFeedForwardSubLayer():
    @abc.abstractmethod
    def forward(self, x:Tensor):
        raise NotImplementedError
    
    @abc.abstractmethod
    def __call__(self, x:Tensor):
        raise NotImplementedError

class IAttentionSubLayer():
    @abc.abstractmethod
    def forward(self, xq : Tensor, xk : Tensor, xv : Tensor, recurrent_memory : Optional[Tensor] = None):
        raise NotImplementedError
    
    @abc.abstractmethod
    def __call__(self, xq : Tensor, xk : Tensor, xv : Tensor, recurrent_memory : Optional[Tensor] = None):
        raise NotImplementedError

class Identity(nn.Identity):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def forward(self, x, *args, **kwargs):
        return x
