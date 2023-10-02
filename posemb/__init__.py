from typing import Any, Optional, Tuple, List, Iterable

import torch
import torch.nn as nn
from torch import Tensor

import posemb.interface

class LearnedPositionalEmbedding(posemb.interface.IPositionalEmbedding):
    def __init__(self, sequence_length : int, d_model : int):
        self.positional_embedding = nn.Embedding(sequence_length, d_model)

    def forward(self, x : Tensor):
        B, T, C = x.size()
        return x + self.positional_embedding(torch.arange(T))

class SinPositionalEmbedding(posemb.interface.IPositionalEmbedding):
    cache = None

    def __init__(self, sequence_length : int, d_model : int):
        super().__init__()

        if SinPositionalEmbedding.cache is None:
            # we flip here so that in case the user wants to extend the sequence length, tokens in the original zone end up with the same values
            indices = torch.arange(sequence_length).flip(-1).unsqueeze(1)
            freqs = 1e-4 ** (torch.linspace(0, -1, d_model//2)) # frequencies from 1.0 ... 1e-4
            angles = (indices * freqs)
            s, c = angles.sin(), angles.cos()
            # interleave sines and cosines
            self.register_buffer('positional_embedding', torch.stack((s, c), dim=-1).flatten(-2)) # (T, C)

            SinPositionalEmbedding.cache = self

    def forward(self, x : Tensor):
        B, T, C = x.size()

        cache = SinPositionalEmbedding.cache
        return x + cache.positional_embedding[-T:,:]

def rot2d_interleaved(cos, sin, t):
    # creates tensor with perpendicular (x,y) -> (-y,x) taken of last dimension, if you treat every two consecutive entries as (x,y)
    # e.g. [1,2,3,4] -> [-2,1,-4,3]
    t_perp = torch.stack((-t[..., 1::2], t[..., 0::2]), dim=-1).flatten(-2)

    return (t * cos[:t.size(-2), :]) + (t_perp * sin[:t.size(-2), :])

def rotary_embedding(Q, T):
    indices = torch.arange(T).flip(-1).reshape(-1, 1)
    angular_velocity = 1e-4 ** torch.linspace(0, -1, Q//2) # frequencies from 1.0 ... 1e-4
    angular_velocity = angular_velocity.repeat_interleave(2, -1)
    angles = indices * angular_velocity
    return angles.sin(), angles.cos()

class RotaryEmbedding(nn.Module, posemb.interface.IQueryKeyEmbedding):
    # using first instance as a cache so we don't waste memory by duplicating our registered buffers per layer
    cache = None
    def __init__(self, sequence_length : int, d_query : int):
        super().__init__()

        if RotaryEmbedding.cache is None:
            sin, cos = rotary_embedding(d_query, sequence_length)
            self.register_buffer('sin', sin)
            self.register_buffer('cos', cos)
            RotaryEmbedding.cache = self

    def forward(self, x : Tuple[Tensor, Tensor]):
        q, k = x
        cache = RotaryEmbedding.cache
        cos, sin = cache.cos, cache.sin
        return rot2d_interleaved(cos, sin, q), rot2d_interleaved(cos, sin, k)

class XPosEmbedding(nn.Module, posemb.interface.IQueryKeyEmbedding):
    def __init__(self, sequence_length : int, d_query : int):
        super().__init__()
        Q = d_query

        self.rotary = RotaryEmbedding(sequence_length=sequence_length, d_query=d_query)

        seq_indices = torch.arange(sequence_length,).float()
        d_embed_indices = torch.arange(0, Q, 2).float()
        scale_base : float = 512
        embed_index_scales = (d_embed_indices + 0.4 * Q) / (1.4 * Q) # hardcoded zeta for each embedding channel
        remapped_seq_positions = (seq_indices - sequence_length // 2) / scale_base
        # multiplying a horizontal embed_index_scales vector (via exponentiation) with a vertical set of remapped sequence indices [-1/256...1/256]
        scale = embed_index_scales ** remapped_seq_positions.reshape(-1, 1) # (T, F)
        scale = scale.repeat_interleave(2, -1) # (T, E)
        self.register_buffer('scale', scale) # (T, E)

    def forward(self, x : Tuple[Tensor, Tensor]):
        q, k = self.rotary(x)
        return q * self.scale, k / self.scale
