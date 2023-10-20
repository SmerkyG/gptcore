from util.config import Factory

from typing import Callable, Any, Optional, Tuple, List, Iterable

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

import model.interface
from model.hparams import HParams

import model.core
import posemb.interface

class GPT2FeedForwardSubLayer(model.core.TransformerLayerPart, model.interface.IFeedForwardSubLayer):
    def __init__(self, hidden_activation_factory : Callable[..., nn.Module] = Factory(nn.GELU, approximate='tanh')):
        super().__init__()
        hparams = self.hparams
        d_hidden = int(hparams.d_model * hparams.feedforward_d_model_ratio)
        self.ff_expansion = nn.Linear(hparams.d_model, d_hidden, bias=False)
        self.activation = hidden_activation_factory()
        self.ff_contraction = nn.Linear(d_hidden, hparams.d_model, bias=False)
        self.dropout = nn.Dropout(hparams.dropout)

    def forward(self, x : Tensor):
        x = self.ff_expansion(x)
        x = self.activation(x)
        x = self.ff_contraction(x)
        x = self.dropout(x)
        return x

class GPT2AttentionSubLayer(model.core.TransformerLayerPart, model.interface.IAttentionSubLayer):
    def __init__(self, attention_factory : Callable[..., model.core.IAttention] = Factory(model.core.TorchAttention)):
        super().__init__()
        hparams, layer_id = self.hparams, self.layer_id
        C = hparams.d_model
        H = hparams.n_head
        KVH = int(hparams.n_head * hparams.n_kv_head_ratio)
        K = int(hparams.d_qk_ratio * hparams.d_model / hparams.n_head)
        Q = int(hparams.d_qk_ratio * hparams.d_model / hparams.n_head)
        V = int(hparams.d_v_ratio * hparams.d_model / hparams.n_head)
        T = hparams.max_sequence_length
        
        self.w_query = nn.Linear(C, H * Q, bias=False)
        self.w_key = nn.Linear(C, KVH * K, bias=False)
        self.w_value = nn.Linear(C, KVH * V, bias=False)

        self.w_out = nn.Linear(H * V, C, bias=False)
        def w_out_init(m): m.weight *= ((2 * self.hparams.n_layer)**-0.5)
        model.core.defer_module_init(self.w_out, w_out_init)

        self.rotary_positional_embedding = hparams.rotary_positional_embedding_factory(T, Q)

        self.attention_module = attention_factory()

    def forward(self, xq : Tensor, xk : Tensor, xv : Tensor, recurrent_memory : Optional[Tensor] = None):       
        hparams = self.hparams
        B, T, C = xq.size() # batch size, sequence length, token embedding dimension count
        H = hparams.n_head
        KVH = int(hparams.n_head * hparams.n_kv_head_ratio)
        Q = int(hparams.d_qk_ratio * hparams.d_model / hparams.n_head)
        K = int(hparams.d_qk_ratio * hparams.d_model / hparams.n_head)
        V = int(hparams.d_v_ratio * hparams.d_model / hparams.n_head)

        q = self.w_query(xq) # (B, T, H*Q)
        k = self.w_key(xk) # (B, T, H*K)
        v = self.w_value(xv) # (B, T, H*V)

        # transpose H and T dimensions so that all heads can be worked on together with a single matrix (required by all efficient attention/retention implementations)
        q = q.view(B, T, H, Q).transpose(1, 2) # (B, H, T, Q)
        k = k.view(B, T, KVH, K).transpose(1, 2) # (B, H, T, K)
        v = v.view(B, T, KVH, V).transpose(1, 2) # (B, H, T, V)

        # rotate queries and keys via RoPE / XPos
        q, k = self.rotary_positional_embedding((q, k))

        # support for grouped-query attention
        # if there are fewer k/v heads than total heads, repeat them until the number matches
        if KVH < H:
            reps = H // KVH
            k = k[:,:,None,:,:].expand(B, KVH, reps, T, K).contiguous().view(B, H, T, K)
            v = v[:,:,None,:,:].expand(B, KVH, reps, T, V).contiguous().view(B, H, T, V)

        y = self.attention_module(q, k, v, recurrent_memory) # (B, H, T, V)

        # squeeze all the head embeddings together into a single dimension
        y = y.transpose(1, 2).contiguous().view(B, T, H*V) # (B, H, T, V) -> (B, T, H*V)

        # project the result to our model embedding size
        return self.w_out(y) # (B, T, H*V) -> (B, T, C)
