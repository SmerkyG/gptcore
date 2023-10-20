import abc

from util.config import Factory

from typing import Any, Optional, Tuple, List, Iterable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class IBiasMask():
    @abc.abstractmethod
    def forward(self, q:Tensor):
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, q:Tensor):
        raise NotImplementedError

class IMulMask():
    @abc.abstractmethod
    def forward(self, q:Tensor):
        raise NotImplementedError
    
    @abc.abstractmethod
    def __call__(self, q:Tensor):
        raise NotImplementedError

def causal_mul_mask(T):
    mask = torch.ones(T, T)
    mask = mask.masked_fill(not mask.tril(), float('-inf')) # (T, T)
    return mask

def causal_bias_mask(T):
    return torch.full((T, T), float('-inf')).triu(1)

class NoBiasMask(nn.Module, IBiasMask):
    def __init__(self, block_size : int, n_heads : int, layer_id : int):
        super().__init__()

    def forward(self, q:Tensor):
        return 0.0

class NoMulMask(nn.Module, IMulMask):
    def __init__(self, block_size : int, n_heads : int, layer_id : int):
        super().__init__()

    def forward(self, q:Tensor):
        return 1.0

class CausalMulMask(nn.Module, IMulMask):
    # using first instance as a cache so we don't waste memory by duplicating our registered buffers per layer
    cache = None
    def __init__(self, block_size : int, n_heads : int, layer_id : int):
        super().__init__()
        if CausalMulMask.cache is None:
            CausalMulMask.cache = self
            T = block_size
            self.register_buffer('mask', causal_mul_mask(T))

    def forward(self, q:Tensor):
        return CausalMulMask.cache.mask

class CausalBiasMask(nn.Module, IBiasMask):
    # using first instance as a cache so we don't waste memory by duplicating our registered buffers per layer
    cache = None
    def __init__(self, block_size : int, n_heads : int, layer_id : int):
        super().__init__()
        if CausalBiasMask.cache is None:
            CausalBiasMask.cache = self
            T = block_size
            self.register_buffer('mask', causal_bias_mask(T))

    def forward(self, q:Tensor):
        return CausalBiasMask.cache.mask

def alibi_mask(T, H):
    bias = (torch.arange(T)[None, :] - torch.arange(T)[:, None]).float() # (T, T)
    bias = bias + causal_bias_mask(T) # (T, T)
    bias = bias.expand(H, -1, -1) # (H, T, T)
    head_bias_slopes = (2 ** torch.linspace(-8.0/H, -8.0, H)).unsqueeze(-1).unsqueeze(-1) # (H, 1, 1)
    bias = bias * head_bias_slopes # (H, T, T)
    return bias

class AlibiMask(nn.Module, IBiasMask):
    # using first instance as a cache so we don't waste memory by duplicating our registered buffers per layer
    cache = None
    def __init__(self, block_size : int, n_heads : int, layer_id : int):
        super().__init__()
        if AlibiMask.cache is None:
            AlibiMask.cache = self
            T = block_size
            H = n_heads
            self.register_buffer('mask', alibi_mask(T, H))

    def forward(self, q:Tensor):
        return AlibiMask.cache.mask
