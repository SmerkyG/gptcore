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

def rwkv5_2_recurrent(r_in, k_in, v_in, w_in, u, kv_state):
    L = r_in.size(-2)
    out = []
    for t in range(L):
        r, k, v, w = r_in[...,t:t+1,:], k_in[...,t:t+1,:], v_in[...,t:t+1,:], w_in[...,t:t+1,:]
        kv = k.mT @ v # KV
        out.append( r @ (kv_state + u.mT * kv) ) # 1K @ (KV + 1)
        kv_state = (w.mT * kv_state) + kv # KV
    out = torch.cat(out, dim=-2)
    return out, kv_state

def sanity_check():
    T = 6
    B = 1
    H = 1
    K,V = 3,5
    r = torch.rand(B,H,T,K)
    k = torch.rand(B,H,T,K)
    v = torch.rand(B,H,T,V)
    w = torch.rand(B,H,T,K)
    u = torch.rand(1,H,1,K)
    kv_state = torch.zeros(B,H,K,V)

    precision_dtype, precision_min_val = torch.float32, 0.02 # good for fp32 
    #precision_dtype, precision_min_val = torch.float64, 1e-10 # good for fp64   
    w = w.clamp(precision_min_val)

    # recurrent
    out, _ = rwkv5_2_recurrent(r,k,v,w,u,kv_state)
    print(out)

    # parallel
    out, _ = rwkv_inner(r,k,v,w,u,kv_state,chunk_len=3)
    print(out)

if __name__ == "__main__":
    sanity_check()
    exit()

from util.config import Factory

import posemb.interface

import model.interface
import model.core
from model.hparams import HParams

from model.rwkv import RWKVConfig

class RWKV5_2_AttentionSubLayer(model.core.TransformerLayerPart, model.interface.IAttentionSubLayer):
    def __init__(self):
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

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_mix_g = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

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

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_kv_head, self.k_head_size)) # (KVH, K)

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(args.n_embd, self.n_head * self.r_head_size, bias=False)
        self.key = nn.Linear(args.n_embd, self.n_kv_head * self.k_head_size, bias=False)
        self.value = nn.Linear(args.n_embd, self.n_kv_head * self.v_head_size, bias=False)
        self.output = nn.Linear(args.dim_v, args.n_embd, bias=False)
        self.gate = nn.Linear(args.n_embd, args.dim_v, bias=False)

        self.rotary_positional_embedding = hparams.rotary_positional_embedding_factory(hparams.max_sequence_length, int(hparams.d_qk_ratio * hparams.d_model / hparams.n_head))

        self.ln_x = nn.GroupNorm(self.n_kv_head, args.dim_v)

    def post_init_fn(self, myself):
        zero = [self.output]
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

        # 24 is optimal chunk length (longer will use too much memory and cause precision problems or even numerical instability, shorter is inefficient)
        chunk_len = 24

        # padding to support fast path for non-exact chunk size multiple sequence lengths        
        n_padding = (chunk_len - x.size(-2) % chunk_len) % chunk_len
        if n_padding != 0:
            x = F.pad(x, [0, 0, 0, n_padding, 0, 0])

        H = self.n_head
        KVH = self.n_kv_head
        R = self.r_head_size
        K = self.k_head_size
        V = self.v_head_size

        B, T, C = x.size()

        xx = self.time_shift(x) # Mix x with the previous timestep to produce kx, vx, rx, gx
        kx = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        vx = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        rx = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        gx = x * self.time_mix_g + xx * (1 - self.time_mix_g)

        r = self.receptance(rx).view(B, T, H, K).transpose(1, 2) # BHTK
        k = self.key(kx).view(B, T, KVH, K).transpose(1, 2)      # BHTK
        v = self.value(vx).view(B, T, KVH, V).transpose(1, 2)    # BHTV
        g = F.silu(self.gate(gx))
        
        r, k = self.rotary_positional_embedding((r, k))

        # support for grouped-query attention
        # if there are fewer k/v heads than total heads, repeat them until the number matches
        time_decay = self.time_decay.float() # (KVH,K)
        time_faaaa = self.time_faaaa.float() # (KVH,K)
        if KVH < H:
            reps = H // KVH
            k = k[:,:,None,:,:].expand(B, KVH, reps, T, K).contiguous().view(B, H, T, K)
            v = v[:,:,None,:,:].expand(B, KVH, reps, T, V).contiguous().view(B, H, T, V)
            time_decay = time_decay.expand(reps, KVH, K).contiguous().view(H, K)
            time_faaaa = time_faaaa.expand(reps, KVH, K).contiguous().view(H, K)

        kv_state = recurrent_memory
        if kv_state is None:
            kv_state = torch.zeros(B, H, K, V, device=r.device, dtype=r.dtype)  # state

        if kv_state.dtype != r.dtype:
            kv_state = kv_state.contiguous().to(r.dtype) 

        w = torch.exp(-torch.exp(time_decay)).view(1,H,1,K).expand(1,H,T,K)
        u = time_faaaa.float().view(1,H,1,K)
        out, s = rwkv_inner(r, k, v, w, u, kv_state, chunk_len)

        out = out.transpose(1,2).reshape(B*T, H*V)
        out = self.ln_x(out / self.args.head_size_divisor).view(B, T, H*V)

        out = self.output(out * g)

        if n_padding != 0:
            out = out[..., :-n_padding, :] # BTC

        return out
