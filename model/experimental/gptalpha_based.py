from typing import Callable, Any, Optional, Tuple, List, Iterable, Callable

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

def taylor_exp(x: Tensor):
    ones = torch.ones(x[..., :1].shape).to(x.device)
    x2 = (x.unsqueeze(-1) * x.unsqueeze(-2)).flatten(-2) / math.sqrt(2)
    return torch.cat([ones, x, x2], dim=-1)

def gptAB_recurrent(q_in, k_in, v_in, w, kv_state = None, eps=1e-12):
    K = k_in.size(-1)
    V = v_in.size(-1)
    if kv_state is None:
        kv_state = torch.zeros(1+K+K*K, V, dtype=q_in.dtype, device=q_in.device)
    L = q_in.size(-2)
    out = []
    for t in range(L):
        q, k, v = q_in[t:t+1], k_in[t:t+1], v_in[t:t+1]
        q, k = taylor_exp(q), taylor_exp(k)
        kv_state = (w * kv_state) + k.mT @ v  # (1+K+K*K)V
        out.append(q @ kv_state)
    out = torch.cat(out, dim=-2)

    return out, kv_state

def gptAB_parallel(q, k, v, w, kv_state = None, eps=1e-12):
    K = k.size(-1)
    V = v.size(-1)
    if kv_state is None:
        kv_state = torch.zeros(1+K+K*K, V, dtype=q.dtype, device=q.device)
    T = q.size(-2)
    dt = (torch.arange(T, device=q.device)[:, None] - torch.arange(T, device=q.device)[None, :]).tril() # NOTE - tril is important to not break pow by causing infinities
    w = w.pow(dt)
    w = w.tril() # causality

    attn = q @ k.mT
    attn = 1 + attn + 0.5 * attn.square() # taylor series approximation to exp
    attn = (attn * w).to(q.dtype)

    # NOTE - we may eventually want denominator, a la rwkv4
    #attn = attn / attn.sum(-1, keepdim=True).clamp(eps)
    out = attn @ v

    # FIXME - calc kv_state change
    return out, kv_state


def sanity_check():
    T = 4
    K,V = 3,5
    w = torch.rand(1)
    q = torch.rand(T,K)
    k = torch.rand(T,K)
    v = torch.rand(T,V) #torch.arange(T*V).view(T,V).float()
    kv_state = None# torch.zeros(1+K+K*K,V)

    #w = torch.exp(-torch.exp(time_decay.float())).reshape(1,H,1,1)

    out, _ = gptAB_recurrent(q,k,v,w,kv_state)
    print(out)
    print()

    out, _ = gptAB_parallel(q,k,v,w,kv_state)
    print(out)
    print()

if __name__ == "__main__":
    sanity_check()
    exit()


import model.interface
import model.core

class ApproximatedDecayedAttention(model.core.TransformerLayerPart, model.core.IAttention):
    def __init__(self):
        super().__init__()
        hparams = self.hparams
        H = hparams.n_head

        self.eps = 1e-12

        layer_id = self.layer_id
        with torch.no_grad():
            ratio_0_to_1 = layer_id / max(hparams.n_layer - 1, 1)  # 0 to 1
            ddd = torch.ones(1, 1, hparams.d_model)
            for i in range(hparams.d_model):
                ddd[0, 0, i] = i / hparams.d_model

            # fancy time_decay
            decay_speed = torch.ones(H)
            for h in range(H):
                decay_speed[h] = -8 + 7 * (h / max(H - 1, 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)

    def forward(self, q:Tensor, k:Tensor, v:Tensor, recurrent_memory : Optional[Tensor] = None):
        B,H,T,Q = q.shape

        hparams = self.hparams
        H = hparams.n_head

        w = torch.exp(-torch.exp(self.time_decay.float())).reshape(1,H,1,1)

        out, s = gptAB_parallel(q, k, v, w)
        return out


