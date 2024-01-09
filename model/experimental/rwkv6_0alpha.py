# This file is modified from https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v4neo/src/model.py and is separately licensed according to the following license:
"""
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
"""

########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

from typing import Callable, Any, Optional, Tuple, List, Iterable, Callable

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from .rwkv_inner import rwkv_inner

from util.config import Factory

import posemb.interface

import model.interface
import model.core
from model.hparams import HParams


from model.rwkv import RWKVConfig

import norm

class RWKV6_0_Alpha_AttentionSubLayer(model.core.TransformerLayerPart, model.interface.IAttentionSubLayer):
    def __init__(self, qkv_norm_factory : Callable = Factory(norm.RMSNorm, weight_scaling=False), ):
        super().__init__()

        hparams, layer_id = self.hparams, self.layer_id

        args = RWKVConfig(hparams)

        self.args = args
        self.layer_id = layer_id
        self.ctx_len = args.ctx_len
        self.n_embd = args.n_embd

        self.n_head = args.n_head
        self.n_kv_head = args.n_kv_head
        self.r_head_size = args.dim_rk // args.n_head
        self.k_head_size = args.dim_rk // args.n_head
        self.v_head_size = args.dim_v // args.n_head
        assert args.dim_rk % self.n_head == 0
        assert args.dim_rk % self.n_kv_head == 0
        assert args.dim_v % self.n_kv_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / max(args.n_layer - 1, 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            self.x_maa = nn.Parameter(1 - torch.pow(ddd, ratio_1_to_almost0))
            self.r_maa = nn.Parameter(1 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.w_maa = nn.Parameter(1 - torch.pow(ddd, ratio_1_to_almost0))
            self.k_maa = nn.Parameter(1 - torch.pow(ddd, ratio_1_to_almost0))
            self.v_maa = nn.Parameter(1 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.g_maa = nn.Parameter(1 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            TIME_MIX_EXTRA_DIM = 32
            self.tm_w1 = nn.Parameter(torch.empty(args.n_embd, TIME_MIX_EXTRA_DIM * 5).uniform_(-0.01, 0.01))
            self.tm_w2 = nn.Parameter(torch.zeros(5, TIME_MIX_EXTRA_DIM, args.n_embd))
            W_MIX_EXTRA_DIM = 64
            self.td_w1 = nn.Parameter(torch.empty(args.n_embd, W_MIX_EXTRA_DIM).uniform_(-0.01, 0.01))
            self.td_w2 = nn.Parameter(torch.zeros(W_MIX_EXTRA_DIM, args.n_embd))

            # fancy time_decay
            k_dim_att = args.n_kv_head * self.k_head_size
            decay_speed = torch.ones(k_dim_att)
            for n in range(k_dim_att):
                decay_speed[n] = -6 + 5 * (n / max(k_dim_att - 1, 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(self.n_kv_head, self.k_head_size)) # (KVH, K)
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            tmp = torch.zeros(k_dim_att)
            for n in range(k_dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / max(k_dim_att - 1, 1))) + zigzag

            self.time_first = nn.Parameter(tmp.reshape(self.n_kv_head, self.k_head_size)) # (KVH, K)

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(args.n_embd, self.n_head * self.r_head_size, bias=False)
        self.key = nn.Linear(args.n_embd, self.n_kv_head * self.k_head_size, bias=False)
        self.value = nn.Linear(args.n_embd, self.n_kv_head * self.v_head_size, bias=False)
        self.output = nn.Linear(args.dim_v, args.n_embd, bias=False)
        self.gate = nn.Linear(args.n_embd, args.dim_v, bias=False)

        self.rotary_positional_embedding = hparams.rotary_positional_embedding_factory(hparams.max_sequence_length, int(hparams.d_qk_ratio * hparams.d_model / hparams.n_head))

    def post_init_fn(self, myself):
        zero = [self.receptance, self.key, self.output]
        for m in zero:
            nn.init.zeros_(m.weight)
        # FIXME - init ln_x with something like layer_scale * 0.7
        ortho = [self.value, self.gate]
        for m in ortho:
            if m.weight.shape[0] > m.weight.shape[1]:
                gain = math.sqrt(m.weight.shape[0] / m.weight.  shape[1])
            else:
                gain = 1.0
            nn.init.orthogonal_(m.weight, gain=gain)

    def forward(self, xq : Tensor, xk : Tensor, xv : Tensor, recurrent_memory : Optional[Tensor] = None):
        x = xq # FIXME - support encoder-decoder models

        H = self.n_head
        KVH = self.n_kv_head
        R = self.r_head_size
        K = self.k_head_size
        V = self.v_head_size

        B, T, C = x.size()

        xx = x
        sx = self.time_shift(x) - xx
        #sx = torch.cat((sx.unsqueeze(0), xx[:-1,:])) - xx

        xxx = xx + sx * self.x_maa
        xxx = torch.tanh(xxx @ self.tm_w1).view(B*T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.tm_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        wx = xx + sx * (self.w_maa + mw)
        kx = xx + sx * (self.k_maa + mk)
        vx = xx + sx * (self.v_maa + mv)
        rx = xx + sx * (self.r_maa + mr)
        gx = xx + sx * (self.g_maa + mg)

        r = self.receptance(rx).view(B, T, H, K).transpose(1, 2) # BHTK
        k = self.key(kx).view(B, T, KVH, K).transpose(1, 2)      # BHTK
        v = self.value(vx).view(B, T, KVH, V).transpose(1, 2)    # BHTV
        g = self.gate(gx)

        # rotate queries and keys via RoPE / XPos
        r, k = self.rotary_positional_embedding((r, k))

        # support for grouped-query attention
        # if there are fewer k/v heads than total heads, repeat them until the number matches
        time_decay = self.time_decay.float() # (KVH,K)
        time_first = self.time_first.float() # (KVH,K)
        if KVH < H:
            reps = H // KVH
            k = k[:,:,None,:,:].expand(B, KVH, reps, T, K).contiguous().view(B, H, T, K)
            v = v[:,:,None,:,:].expand(B, KVH, reps, T, V).contiguous().view(B, H, T, V)
            time_decay = time_decay.expand(reps, KVH, K).contiguous().view(H, K)
            time_first = time_first.expand(reps, KVH, K).contiguous().view(H, K)

        kv_state = recurrent_memory
        if kv_state is None:
            kv_state = torch.zeros(B, H, K, V, device=r.device, dtype=r.dtype)

        if r.dtype == torch.bfloat16 and kv_state.dtype != torch.bfloat16:
            kv_state = kv_state.contiguous().to(torch.bfloat16)        

        w = time_decay.view(1, H, 1, K)
        w = w + (torch.tanh(wx @ self.td_w1) @ self.td_w2).view(B, H, T, K) # BHTK
        w = torch.exp(-torch.exp(w))

        u = time_first.view(1, H, 1, K)

        out, kv_state = rwkv_inner(r, k, v, w, u, kv_state)

        # TransNormer style normalization after multiplying qkv (same as separate groupnorm of each head but w/o a scaling parameter)
        # normalize each head
        out = norm.Norm.F(out)

        # squeeze all the head embeddings together into a single dimension
        out = out.transpose(1, 2).reshape(B, T, H*V) # (B, H, T, V) -> (B, T, H*V)

        out = self.output(out * g)

        return out
