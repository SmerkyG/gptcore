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

# from PaLM paper (section 5)
class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)
        
class RWKV5_1_AttentionSubLayer(model.core.TransformerLayerPart, model.interface.IAttentionSubLayer):
    def __init__(self, chunk_len : int = 512):
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

            # fancy time_decay
            decay_speed = torch.ones(self.n_kv_head)
            for h in range(self.n_kv_head):
                decay_speed[h] = -8 + 7 * (h / max(self.n_kv_head - 1, 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed) # (KVH)
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())
            
            tmp = torch.zeros(self.n_kv_head)
            for h in range(self.n_kv_head):
                tmp[h] = ratio_0_to_1 * (1 - (h / max(self.n_kv_head - 1, 1)))
            self.time_faaaa = nn.Parameter(tmp) # (KVH)

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(args.n_embd, self.n_head * self.r_head_size, bias=False)
        self.key = nn.Linear(args.n_embd, self.n_kv_head * self.k_head_size, bias=False)
        self.value = nn.Linear(args.n_embd, self.n_kv_head * self.v_head_size, bias=False)
        self.output = nn.Linear(args.dim_v, args.n_embd, bias=False)
        self.gate = nn.Linear(args.n_embd, args.dim_v, bias=False)

        self.rotary_positional_embedding = hparams.rotary_positional_embedding_factory(hparams.max_sequence_length, int(hparams.d_qk_ratio * hparams.d_model / hparams.n_head))

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

        r = self.receptance(rx).view(B, T, H, K).transpose(1, 2) # BHTK
        k = self.key(kx).view(B, T, KVH, K).transpose(1, 2)      # BHTK
        v = self.value(vx).view(B, T, KVH, V).transpose(1, 2)    # BHTV
        g = F.silu(self.gate(gx))
        
        r, k = self.rotary_positional_embedding((r, k))

        # support for grouped-query attention
        # if there are fewer k/v heads than total heads, repeat them until the number matches
        time_decay = self.time_decay.float() # (KVH)
        time_faaaa = self.time_faaaa.float() # (KVH)
        if KVH < H:
            reps = H // KVH
            k = k[:,:,None,:,:].expand(B, KVH, reps, T, K).contiguous().view(B, H, T, K)
            v = v[:,:,None,:,:].expand(B, KVH, reps, T, V).contiguous().view(B, H, T, V)
            time_decay = time_decay.expand(reps, KVH).contiguous().view(H)
            time_faaaa = time_faaaa.expand(reps, KVH).contiguous().view(H)

        k = k.transpose(-2, -1) # BHKT

        s = recurrent_memory
        if s is None:
            s = torch.zeros(B, H, K, V, device=r.device, dtype=r.dtype)  # state

        if r.dtype == torch.bfloat16 and s.dtype != torch.bfloat16:
            s = s.contiguous().to(torch.bfloat16)

        L = T
        T = self.chunk_len

        if L == 1:
            t_first = torch.exp(-torch.exp(time_decay)).unsqueeze(-1).unsqueeze(-1) # (H, 1, 1)
            t_decay = time_faaaa.unsqueeze(-1).unsqueeze(-1) # (H, 1, 1)
            out = torch.zeros(B, H, L, V, device=r.device, dtype=r.dtype) # output
            for t in range(L):
                rt = r[...,t:t+1,:]
                kt = k[...,:,t:t+1]
                vt = v[...,t:t+1,:]
                at = kt @ vt
                out[..., t:t+1, :] = (rt @ (t_first * at + s)).squeeze(1)
                s = at + t_decay * s
        else:
            w = torch.exp(-torch.exp(time_decay)).unsqueeze(-1) # (H,1)
            u = time_faaaa.unsqueeze(-1) # (H, 1)

            ws = w.pow(T).reshape(1, H, 1, 1)
            ind = torch.arange(T-1, -1, -1, device=r.device).unsqueeze(0).repeat(H, 1) # (H,T)
            w = w.repeat(1, T).pow(ind) # (H,T)

            wk = w.reshape(1, H, 1, T)
            wb = wk.transpose(-2, -1).flip(2) # (1, H, T, 1)

            w = torch.cat([w[:, 1:], u], dim=1) # (H, T)
            w = F.pad(w, (0, T))
            w = torch.tile(w, [T])
            w = w[:, :-T].reshape(-1, T, 2 * T - 1)
            w = w[:, :, T-1:].reshape(1, H, T, T)

            wk = wk.to(dtype=r.dtype)
            wb = wb.to(dtype=r.dtype)
            ws = ws.to(dtype=r.dtype)
            
            out = []
            for i in range(L // T):
                rr = r[:, :, i*T:i*T+T, :]
                kk = k[:, :, :, i*T:i*T+T]
                vv = v[:, :, i*T:i*T+T, :]

                out.append((((rr @ kk) * w) @ vv).to(r.dtype)  +  ((rr @ s) * wb).to(r.dtype))

                s = ws * s + (kk * wk) @ vv
            out = torch.cat(out, dim=-2)
        

        out = out.transpose(1, 2).contiguous().view(B * L, H*V) # BHTS -> BTHS -> BTC
        out = self.ln_x(out / self.args.head_size_divisor).view(B, L, H*V)

        out = self.output(out * g)        
        return out

class RWKV_ChannelMixSubLayer(model.core.TransformerLayerPart, model.interface.IFeedForwardSubLayer):
    def __init__(self):
        super().__init__()
        hparams, layer_id = self.hparams, self.layer_id
        args = RWKVConfig(hparams)
        self.args = args
        self.layer_id = layer_id
        
        with torch.no_grad():  # fancy init of time_mix
            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
        
        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    def post_init_fn(self, myself):
        zero = [self.value, self.receptance]
        for m in zero:
            nn.init.zeros_(m.weight)
        ortho = [self.key]
        for m in ortho:
            if m.weight.shape[0] > m.weight.shape[1]:
                gain = math.sqrt(m.weight.shape[0] / m.weight.shape[1])
            else:
                gain = 1.0
            nn.init.orthogonal_(m.weight, gain=gain)

    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        k = torch.square(torch.relu(k))
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv
