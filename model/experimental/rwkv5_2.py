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

def rwkv5_2_recurrent(s, r_in, k_in, v_in, w_in, u):
    L = r_in.size(-2)
    out = []
    for t in range(L):
        r, k, v = r_in[...,t:t+1,:], k_in[...,t:t+1,:], v_in[...,t:t+1,:]
        w = w_in[...,t:t+1,:,:]
        kv = k.mT @ v # KV
        out.append( r @ (s + u.mT * kv) ) # 1K @ (KV + 1)
        s = (w.mT * s) + kv # KV
    out = torch.cat(out, dim=-2)
    return out, s

def sanity_check():
    T = 6
    B = 1
    H = 1
    K,V = 3,5
    r = torch.rand(B,H,T,K)
    k = torch.rand(B,H,T,K)
    v = torch.rand(B,H,T,V)
    w = torch.ones(B,T,H,K)/math.e #torch.rand(1,1,H,K).expand(B,T,H,K)
    u = torch.rand(  1,H,K)
    s = torch.zeros(B,H,K,V)

    precision_dtype, precision_min_val = torch.float32, 0.02 # good for fp32 
    #precision_dtype, precision_min_val = torch.float64, 1e-10 # good for fp64   
    w = w.clamp(precision_min_val)

    # recurrent
    out, _ = rwkv5_2_recurrent(s,r,k,v,w,u)
    print(out)

    # parallel
    out, _ = rwkv_inner(s,r,k,v,w,u,chunk_len=3)
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
    def __init__(self, rotary_positional_embedding_factory : Callable[..., posemb.interface.IQueryKeyEmbedding | nn.Identity] = Factory(nn.Identity)):
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

        #self.time_mixer_k = model.core.DataDependentTimeLerp()
        #self.time_mixer_v = model.core.DataDependentTimeLerp()
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(args.n_embd, self.n_head * self.r_head_size, bias=False)
        self.key = nn.Linear(args.n_embd, self.n_kv_head * self.k_head_size, bias=False)
        self.value = nn.Linear(args.n_embd, self.n_kv_head * self.v_head_size, bias=False)
        self.output = nn.Linear(args.dim_v, args.n_embd, bias=False)
        self.gate = nn.Linear(args.n_embd, args.dim_v, bias=False)

        self.rotary_positional_embedding = rotary_positional_embedding_factory()

        self.ln_x = nn.GroupNorm(self.n_kv_head, args.dim_v)

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

        xx = self.time_shift(x) # Mix x with the previous timestep to produce kx, vx, rx, gx
        kx = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        vx = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        rx = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        gx = x * self.time_mix_g + xx * (1 - self.time_mix_g)
        # Mix kx, vx with the previous timestep in a learned manner to produce new time-mixed xk, xv
        #kx = self.time_mixer_k(kx)
        #vx = self.time_mixer_v(vx)
        #rx = gx = x

        r = self.receptance(rx).view(B, T, H, K).transpose(1, 2) # BTHK
        k = self.key(kx).view(B, T, KVH, K).transpose(1, 2)      # BTHK
        v = self.value(vx).view(B, T, KVH, V).transpose(1, 2)    # BTHV
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

        s = recurrent_memory
        if s is None:
            s = torch.zeros(B, H, K, V, device=r.device, dtype=r.dtype)  # state

        if r.dtype == torch.bfloat16 and s.dtype != torch.bfloat16:
            s = s.contiguous().to(torch.bfloat16)        

        w = torch.exp(-torch.exp(time_decay)).unsqueeze(0).expand(1,T,H,K)
        u = time_faaaa.float().unsqueeze(0) # (1,H,K)
        out, s = rwkv_inner(s, r, k, v, w, u)

        out = out.reshape(B*T, H*V)
        out = self.ln_x(out / self.args.head_size_divisor).view(B, T, H*V)

        out = self.output(out * g)
        return out
