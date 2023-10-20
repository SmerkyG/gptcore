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
        self.n_kv_head = int(hparams.n_head * hparams.n_kv_head_ratio)
        self.n_layer=hparams.n_layer
        self.dim_ffn=int(hparams.feedforward_d_model_ratio * hparams.d_model)
        self.dim_rk=int(hparams.d_qk_ratio * hparams.d_model)
        self.dim_v=int(hparams.d_v_ratio * hparams.d_model)
        self.ctx_len=hparams.max_sequence_length
        self.head_size_divisor=8

class RWKV5r5_AttentionSubLayer(model.core.TransformerLayerPart, model.interface.IAttentionSubLayer):
    def __init__(self, chunk_len : int = 64, rotary_positional_embedding_factory : Callable[..., posemb.interface.IQueryKeyEmbedding | nn.Identity] = Factory(nn.Identity)):
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

        self.chunk_len = chunk_len
        assert self.ctx_len % self.chunk_len == 0

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

            # FIXME - below code is for when rwkv5 changed to use bigger embedding sized decay rates, not one entry just per head, but not sure how to integrate it
            # FIXME - not sure this can really work when dim_rk != dim_v... doesn't some decay apply to the v dimension of the KxV state?

            # fancy time_decay
            v_dim_att = args.n_kv_head * self.v_head_size
            decay_speed = torch.ones(v_dim_att)
            for n in range(v_dim_att):
                decay_speed[n] = -6 + 5 * (n / max(v_dim_att - 1, 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(self.n_kv_head, self.r_head_size)) # (H, V)
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            tmp = torch.zeros(v_dim_att)
            for n in range(v_dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / max(v_dim_att - 1, 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_kv_head, self.r_head_size)) # (H, V)      

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
        T = self.chunk_len
        R = self.r_head_size
        K = self.k_head_size
        V = self.v_head_size

        B, TT, C = x.size()

        xx = self.time_shift(x) # Mix x with the previous timestep to produce xk, xv, xr
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        xg = x * self.time_mix_g + xx * (1 - self.time_mix_g)

        r = self.receptance(xr).view(B, TT, self.n_head, self.r_head_size).transpose(1, 2)            # BTC -> BHTS
        k = self.key(xk).view(B, TT, self.n_kv_head, self.k_head_size).transpose(1, 2) # BTC -> BHTS
        v = self.value(xv).view(B, TT, self.n_kv_head, self.v_head_size).transpose(1, 2)                 # BTC -> BHTS
        g = F.silu(self.gate(xg))
        
        r, k = self.rotary_positional_embedding((r, k))

        # support for grouped-query attention
        # if there are fewer k/v heads than total heads, repeat them until the number matches
        time_decay = self.time_decay.float()
        time_faaaa = self.time_faaaa.float()
        if KVH < H:
            reps = H // KVH
            k = k[:,:,None,:,:].expand(B, KVH, reps, TT, K).contiguous().view(B, H, TT, K)
            v = v[:,:,None,:,:].expand(B, KVH, reps, TT, V).contiguous().view(B, H, TT, V)
            time_decay = time_decay.expand(KVH, reps, V).contiguous().view(H, V)
            time_faaaa = time_faaaa.expand(KVH, reps, V).contiguous().view(H, V)

        k = k.transpose(-2, -1) # BHTS -> BHST

        B, H, TT, R = r.size()

        s = recurrent_memory
        if s is None:
            s = torch.zeros(B, H, K, V, device=r.device, dtype=r.dtype)  # state

        if r.dtype == torch.bfloat16 and s.dtype != torch.bfloat16:
            s = s.contiguous().to(torch.bfloat16)
        
        x = torch.zeros(B, H, TT, V, device=r.device, dtype=r.dtype) # output

        if T == 1:
            t_first = torch.exp(-torch.exp(time_decay)).unsqueeze(-1).unsqueeze(-1) # (H, 1, 1)
            t_decay = time_faaaa.unsqueeze(-1).unsqueeze(-1) # (H, 1, 1)
            for t in range(TT):
                rt = r[...,t:t+1,:]
                kt = k[...,:,t:t+1]
                vt = v[...,t:t+1,:]
                at = kt @ vt
                x[..., t:t+1, :] = (rt @ (t_first * at + s)).squeeze(1)
                s = at + t_decay * s
        else:
            w = torch.exp(-torch.exp(time_decay)).unsqueeze(-1) # (H,1) new (H, V, 1)
            u = time_faaaa.float().unsqueeze(-1) # (H, 1) new (H, V, 1)

            ws = w.pow(T).reshape(1, H, 1, V) # FIXME - order?
            ind = torch.arange(T-1, -1, -1, device=r.device).unsqueeze(0).unsqueeze(0).repeat(H, V, 1) # (H,T) new (H,V,T)
            w = w.repeat(1, 1, T) # (H,T) new (H,V,T)
            w = w.pow(ind) # (H,T) new (H,V,T)

            wk = w.transpose(-1, -2).reshape(1, H, V, T) # (1, H, V, T)
            wb = wk.transpose(-2, -1).flip(2) # (1, H, T, V) # FIXME - order?

            w = torch.cat([w[:, :, 1:], u], dim=-1) # (H, T) new (H, V, T)
            w = F.pad(w, (0, T))
            w = torch.tile(w, [T])
            w = w[:, :, :-T].reshape(H, V, T, 2 * T - 1)
            w = w[:, :, :, T-1:].reshape(1, H, V, T, T)

            w = w.to(dtype=r.dtype)
            wk = wk.to(dtype=r.dtype)
            wb = wb.to(dtype=r.dtype)
            ws = ws.to(dtype=r.dtype)
            
            #return self.jit_func_2(r, k, v, w, wk, wb, ws)

            for i in range(TT // T):
                rr = r[:, :, i*T:i*T+T, :] # (B, H, T, R)
                kk = k[:, :, :, i*T:i*T+T] # (B, H, K, T)
                vv = v[:, :, i*T:i*T+T, :] # (B, H, T, V)

                xx = rr @ kk # (B, H, T, T)
                xx = xx.unsqueeze(-3) # (B, H, 1, T, T)
                xx = xx * w # (B, H, V, T, T)
                vv2 = vv.transpose(-1, -2).unsqueeze(-1) # (B, H, V, T, 1)
                xx = xx @ vv2 # (B, H, V, T, 1)
                xx = xx.squeeze(-1).transpose(-1, -2) # (B, H, T, V)
                remnant = (rr @ s) * wb # (B, H, T, V)
                xx = xx + remnant # (B, H, T, V)
                x[:, :, i*T:i*T+T, :] = xx

                s = ws * s # (B, H, K, V)
                s = s + (kk * wk) @ vv
        
        
        x = x.transpose(1, 2).contiguous().view(B * TT, H*V) # BHTS -> BTHS -> BTC
        x = self.ln_x(x / self.args.head_size_divisor).view(B, TT, H*V)

        x = self.output(x * g)        
        return x

    def att_seq_v5_2(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, k_mix, v_mix, r_mix, g_mix, t_decay, t_first, kw, vw, rw, gw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, omx, orx, omy, ory):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)
        gx = xx * g_mix + sx * (1 - g_mix)

        H = t_decay.shape[0]
        S = x.shape[-1] // H
        T = x.shape[0]

        r = (rx @ rw).view(T, H, S).transpose(0, 1)
        k = (kx @ kw).view(T, H, S).transpose(0, 1).transpose(-2, -1)
        v = (vx @ vw).view(T, H, S).transpose(0, 1)
        g = F.silu(gx @ gw)

        out = torch.empty((T, H, S), dtype=r.dtype, device=r.device)
        for t in range(T):
            rt = r[:,t:t+1,:]
            kt = k[:,:,t:t+1]
            vt = v[:,t:t+1,:]
            at = kt @ vt
            out[t] = (rt @ (t_first * at + s)).squeeze(1)
            s = at + t_decay * s

        out = out.reshape(T, H*S)
        out = F.group_norm(out, num_groups=H, weight=lx_w, bias=lx_b)
        out = out.to(dtype=x.dtype) * g
        out = out @ ow

        return x + out, xx[-1,:], s