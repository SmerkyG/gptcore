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

from util.config import Factory

from typing import Callable, Any, Optional, Tuple, List, Iterable, Callable

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

import posemb.interface

import model.interface
import model.core
from model.hparams import HParams


class RWKVConfig():
    def __init__(self, hparams : HParams):
        super().__init__()
        self.n_embd = hparams.d_model
        self.n_head = hparams.n_head
        self.n_layer=hparams.n_layer
        self.dim_ffn=int(hparams.feedforward_d_model_ratio * hparams.d_model)
        self.dim_rk=int(hparams.d_qk_ratio * hparams.d_model)
        self.dim_v=int(hparams.d_v_ratio * hparams.d_model)
        self.ctx_len=hparams.max_sequence_length
        self.head_size_divisor=8

class RWKV5r5_AttentionSubLayer(nn.Module, model.interface.IAttentionSubLayer, model.core.TransformerLayerPart):
    def __init__(self, chunk_len : int = 128, rotary_positional_embedding_factory : Callable[..., posemb.interface.IQueryKeyEmbedding | nn.Identity] = Factory(nn.Identity)):
        super().__init__()

        hparams, layer_id = self.hparams, self.layer_id

        args = RWKVConfig(hparams)

        self.args = args
        self.layer_id = layer_id
        self.ctx_len = args.ctx_len
        self.n_embd = args.n_embd

        self.n_head = args.n_head
        self.rk_head_size = args.dim_rk // args.n_head
        self.v_head_size = args.dim_v // args.n_head
        assert args.dim_rk % self.n_head == 0
        assert args.dim_v % self.n_head == 0

        self.chunk_len = chunk_len
        assert self.ctx_len % self.chunk_len == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_mix_g = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            # FIXME - below code is for when rwkv5 changed to use bigger embedding sized decay rates, not one entry just per head, but not sure how to integrate it
            # FIXME - not sure this can really work when dim_rk != dim_v... doesn't some decay apply to the v dimension of the KxV state?

            # fancy time_decay
            decay_speed = torch.ones(args.dim_rk) 
            for n in range(args.dim_rk):          
                decay_speed[n] = -6 + 5 * (n / (args.dim_rk - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(self.n_head, self.rk_head_size)) # (H, R)      
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            tmp = torch.zeros(args.dim_rk)
            for n in range(args.dim_rk):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_rk - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.rk_head_size)) # (H, R)      

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(args.n_embd, args.dim_rk, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_rk, bias=False)
        self.value = nn.Linear(args.n_embd, args.dim_v, bias=False)
        self.output = nn.Linear(args.dim_v, args.n_embd, bias=False)
        self.gate = nn.Linear(args.n_embd, args.dim_v, bias=False)

        self.rotary_positional_embedding = rotary_positional_embedding_factory()

        self.ln_x = nn.GroupNorm(self.n_head, args.dim_v)

    def forward(self, xq : Tensor, xk : Tensor, xv : Tensor, recurrent_memory : Optional[Tensor] = None):
        x = xq # FIXME - support encoder-decoder models

        H = self.n_head
        T = self.chunk_len
        R = self.rk_head_size

        B, TT, C = x.size()

        xx = self.time_shift(x) # Mix x with the previous timestep to produce xk, xv, xr
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        xg = x * self.time_mix_g + xx * (1 - self.time_mix_g)

        r = self.receptance(xr).view(B, TT, self.n_head, self.rk_head_size).transpose(1, 2)            # BTC -> BHTS
        k = self.key(xk).view(B, TT, self.n_head, self.rk_head_size).transpose(1, 2) # BTC -> BHTS
        v = self.value(xv).view(B, TT, self.n_head, self.v_head_size).transpose(1, 2)                 # BTC -> BHTS
        g = F.silu(self.gate(xg))
        
        r, k = self.rotary_positional_embedding((r, k))

        k = k.transpose(-2, -1) # BHTS -> BHST


        w = torch.exp(-torch.exp(self.time_decay.float())).unsqueeze(-2) # (H,1) new (H, 1, R)
        u = self.time_faaaa.float().unsqueeze(-2) # (H, 1) new(H, 1, R)

        ws = w.pow(T).reshape(1, H, 1, R)
        ind = torch.arange(T-1, -1, -1, device=r.device).unsqueeze(0).unsqueeze(-1).repeat(H, 1, R) # (H,T) new (H,T,R)
        w = w.repeat(1, T, 1) # (H,T) new (H,T,R)
        w = w.pow(ind) # (H,T) new (H,T,R)

        #wk = w.reshape(1, H, 1, T) # (H, T) -> (1, H, 1, T)
        wk = w.transpose(-1, -2).reshape(1, H, R, T) # (1, H, R, T)
        wb = wk.transpose(-2, -1).flip(2)

        w = torch.cat([w[:, 1:, :], u], dim=1) # (H, T+1) new (H, T+1, R)
        w = F.pad(w, (0, 0, 0, T, 0, 0))
        w = torch.tile(w, [1,T,1])
        w = w[:, :-T, :].reshape(-1, T, 2 * T - 1, R)
        w = w[:, :, T-1:].reshape(1, H, T, T, R)

        # w = torch.cat([w[:, 1:], u], dim=1)
        # w = F.pad(w, (0, T))
        # w = torch.tile(w, [T])
        # w = w[:, :-T].reshape(-1, T, 2 * T - 1)
        # w = w[:, :, T-1:].reshape(1, H, T, T)

        w = w.to(dtype=r.dtype)
        wk = wk.to(dtype=r.dtype)
        wb = wb.to(dtype=r.dtype)
        ws = ws.to(dtype=r.dtype)
        
        #return self.jit_func_2(r, k, v, w, wk, wb, ws)
        B, H, TT, RK = r.size()
        V = v.size(-1)

        s = recurrent_memory
        if s is None:
            s = torch.zeros(B, H, RK, V, device=r.device, dtype=r.dtype)  # state

        if r.dtype == torch.bfloat16 and s.dtype != torch.bfloat16:
            s = s.contiguous().to(torch.bfloat16)
        
        x = torch.zeros(B, H, TT, V, device=r.device, dtype=r.dtype) # output


        for i in range(TT // T):
            rr = r[:, :, i*T:i*T+T, :]
            kk = k[:, :, :, i*T:i*T+T]
            vv = v[:, :, i*T:i*T+T, :]

            xx = rr @ kk
            xx = xx * w
            xx = xx @ vv
            xx = xx + (rr @ s) * wb
            x[:, :, i*T:i*T+T, :] = xx

            s = ws * s + (kk * wk) @ vv

            #new_kv = kk @ vv                                # precalc the new kv
            #x[:, :, i*T:i*T+T, :] = rr * (u * new_kv + s)   # output is receptance times a temporary state that is a tweaked new kv plus unmixed old kv
            #s = ws * s + new_kv                             # decay old state and update with new kv
        
        x = x.transpose(1, 2).contiguous().view(B * TT, H*V) # BHTS -> BTHS -> BTC
        x = self.ln_x(x / self.args.head_size_divisor).view(B, TT, H*V)

        x = self.output(x * g)        
        return x

