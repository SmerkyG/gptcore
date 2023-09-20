from typing import Any, Optional, Tuple, List, Iterable

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from util.config import Factory

from . import picogpt

import model.interface
from model.hparams import HParams

class Llama2FeedForwardSubLayer(nn.Module, model.interface.IFeedForwardSubLayer):
    # SwiGLU FFN
    def __init__(self, hparams : HParams, layer_id : int, hidden_activation_factory : Factory = Factory(nn.SiLU)):
        super().__init__()
        d_hidden = int(hparams.d_model * hparams.feedforward_d_model_ratio)
        self.w_hidden = nn.Linear(hparams.d_model, d_hidden, bias=False)
        self.w_gate = nn.Linear(hparams.d_model, d_hidden, bias=False)
        self.activation = hidden_activation_factory()
        self.w_out = nn.Linear(d_hidden, hparams.d_model, bias=False)
        def w_out_init(m): m.weight *= ((2 * self.hparams.n_layer)**-0.5)
        model.core.defer_module_init(self.w_out, w_out_init)
        self.dropout = nn.Dropout(hparams.dropout)

    def forward(self, x : Tensor):
        return self.dropout(self.w_out(self.w_gate(x) * self.activation(self.w_hidden(x))))

class Llama2AttentionSubLayer(nn.Module, model.interface.IAttentionSubLayer):
    def __init__(self, hparams : HParams, layer_id : int, attention_factory : Factory = Factory(core.TorchAttention), n_kv_head : Optional[int] = None):
        super().__init__()
        self.hparams = hparams
        self.n_kv_head = n_kv_head if n_kv_head is not None else hparams.n_head
        base_head_size = int(hparams.d_model / hparams.n_head)
        C = hparams.d_model
        QH = hparams.n_head
        KH = VH = self.n_kv_head
        K = int(hparams.d_qk_ratio * base_head_size)
        Q = int(hparams.d_qk_ratio * base_head_size)
        V = int(hparams.d_v_ratio * base_head_size)

        self.w_query = nn.Linear(C, QH * Q, bias=False)
        self.w_key = nn.Linear(C, KH * K, bias=False)
        self.w_value = nn.Linear(C, VH * V, bias=False)

        self.w_out = nn.Linear(QH * V, C, bias=False)

        self.rotary_positional_embedding = hparams.rotary_positional_embedding_factory(hparams)

        self.attention_module = attention_factory(hparams=hparams, layer_id=layer_id)

    def forward(self, xq : Tensor, xk : Tensor, xv : Tensor, recurrent_memory : Optional[Tensor] = None):       
        hparams = self.hparams
        B, T, C = xq.size() # batch size, sequence length, token embedding dimension count
        QH = hparams.n_head
        KH = VH = self.n_kv_head
        base_head_size = int(hparams.d_model / hparams.n_head)
        K = int(hparams.d_qk_ratio * base_head_size)
        Q = int(hparams.d_qk_ratio * base_head_size)
        V = int(hparams.d_v_ratio * base_head_size)

        q = self.w_query(xq) # (B, T, QH*Q)
        k = self.w_key(xk) # (B, T, KH*K)
        v = self.w_value(xv) # (B, T, VH*V)

        # transpose H and T dimensions so that all heads can be worked on together with a single matrix (required by all efficient attention/retention implementations)
        q = q.view(B, T, QH, Q).transpose(1, 2) # (B, QH, T, Q)
        k = k.view(B, T, KH, K).transpose(1, 2) # (B, KH, T, K)
        v = v.view(B, T, VH, V).transpose(1, 2) # (B, VH, T, V)

        # rotate queries and keys via RoPE / XPos
        q = self.rotary_positional_embedding(q)
        k = self.rotary_positional_embedding(k)

        # support for grouped-query attention
        # if there are fewer k/v heads than total heads, repeat them until the number matches
        if KH < QH:
            reps = QH // KH
            k = k[:,:,None,:,:].expand(B, KH, reps, T, K).contiguous().view(B, QH, T, K)
            v = v[:,:,None,:,:].expand(B, VH, reps, T, V).contiguous().view(B, QH, T, V)

        y = self.attention_module(q, k, v, recurrent_memory) # (B, QH, T, V)

        # squeeze all the head embeddings together into a single dimension
        y = y.transpose(1, 2).contiguous().view(B, T, QH*V) # (B, QH, T, V) -> (B, T, VH*V)

        # project the result to our model embedding size
        return self.w_out(y) # (B, T, QH*V) -> (B, T, C)
