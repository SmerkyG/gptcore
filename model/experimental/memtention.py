from typing import Callable, Any, Optional, Tuple, List, Iterable, Callable

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from util.config import Factory

import model.core
import model.interface

from model.rwkv import RWKVConfig

import norm

"""
motivation for MemTention:

rwkv is already essentially storing a memory of values in its state.. rwkv calls the slot chooser 'keys' but really thats what 'p' is here 
it chooses where to put the values but ALSO where to put the keys, as in REAL keys like the ones in traditional transformers
so in rwkv, you end up with a memory full of values and you use 'r' to select one (or a mush of them)
but here you use q to do traditional attention across all the memory slots full of stored Keys and Values 
just like a transformer would, only this memory is a fixed length and needs no positional information bc its just memory addresses
essentially it bolts on traditional attention to the rwkv memory state, but correctly separating key and value 
so instead of attending to the past as in a transformer, you are attending to the memory
and just like in rwkv, it can intelligently decide what to put in the memory and how long to hold it there
via data driven decays

you can think of that in terms of how transformers work
k and v always refer to the same time entries
q_i dot k_j gives you how MUCH of v_j to use for that k's timeslot j 
attention is meant to simulate a mushy hash table: queries match in a real valued (0.0-1.0) way against keys, and for a given query, you get out the sum of how much each of those keys matches the query times the value at that key's position
"""

    # simplified recurrence
    # sk = (w * sk) + s.mT @ k    # SK
    # sv = (w * sv) + s.mT @ v    # SV
    # out = torch.softmax(q @ sk.mT, dim=-1) @ sv # 1Q @ KS -> 1S, 1S @ SV -> 1V

    # non-decayed simplified recurrence
    # sk = sk + s.mT @ k    # SK
    # sv = sv + s.mT @ v    # SV
    # out = torch.softmax(q @ sk.mT, dim=-1) @ sv # 1Q @ KS -> 1S, 1S @ SV -> 1V

    # non-decayed parallel
    # torch.softmax(q @ torch.cumsum(s.mT @ k, dim=-3).mT, dim=-1) @ torch.cumsum(s.mT @ v, dim=-3) # T1Q @ TKS -> T1S, T1S @ TSV -> T1V
    #=torch.softmax(q @ torch.cumsum(k.mT @ s, dim=-3), dim=-1) @ torch.cumsum(s.mT @ v, dim=-3) # T1Q @ TKS -> T1S, T1S @ TSV -> T1V
    #=torch.softmax((q @ k.mT).tril() @ s, dim=-2).unsqueeze(-2) @ torch.cumsum(s.mT @ v, dim=-3) # TT @ TS -> TS, T1S @ TSV -> T1V
    #=(torch.softmax((q @ k.mT).tril() @ s, dim=-2) @ s.mT).tril() @ v # TT @ TS -> TS, TS @ ST @ TV -> TV

    # 1. o = q @ k.mT          # TQ @ KT -> TT    (weights by output [query] timeslot and key timeslot)
    # 2. o = o.tril() @ s      # TT @ TS -> TS    (weights by output timeslot and key memoryslot)
    # 3. o = softmax(o) @ s.mT # TS @ ST -> TT    (keep output timeslot but map key memoryslot back to a key/value timeslot)
    # 4. o = o.tril() @ v      # TT @ TV -> TV    (apply key/value timeslot weights to values)    

    # this is interesting in terms of interpretation:
    # steps 2 and 3 essentially translate from timeslot to memoryslot and then back to timeslot 'undoing' the operation
    # but importantly applying softmax in the middle, to give us full traditional attention fidelity!
    # its like 
    # @= s    # do
    # @= s.mT # undo
    # if you eliminate steps 2 and 3 you get linear attention!

# 32 is optimal chunk length (longer will use too much memory, shorter is inefficient)
def memtention_inner(q,s,k,v,w,u,skv_state,chunk_len=32):
    """
    expects
    skv_state : (B,H,S,K+V) # recurrent s,k+v state
    q : (B,H,L,K)
    k : (B,H,L,K)
    v : (B,H,L,V)
    w : (B,H,L,S) or (1,H,L,S)
    u : (1,H,S)
    """
    B,H,L,K = k.size()
    V = v.size(-1)
    S = s.size(-1)
    T = chunk_len

    kv = torch.cat([k,v],dim=-1)
    if L == 1:
        skv = s @ kv
        out = q @ (skv_state + u.mT * skv)
        skv_state = w.mT * skv_state + skv
        return out, skv_state
    else:
        # FIXME - support fast path for non-exact multiples
        # ensure it's an exact multiple
        if L % T != 0:
            T = 1

        N = L // T

        # this has to be done to avoid numerical instability (inf/NaN) when w is used as a divisor up to chunk_length//2 places away (so precision_min_val^(T//2) has to be in fp range)
        precision_dtype, precision_min_val = torch.float32, 0.02 # good for fp32 
        #precision_dtype, precision_min_val = torch.float64, 1e-10 # good for fp64
        
        w = w.clamp(precision_min_val)

        # calculate cumulative decay in log space where it won't overflow
        w_log = w.float().log() # (1,H,L,S) or (B,H,L,S)

        # prepend a zero to make it easy to get shifted version
        w_log = torch.cat([torch.zeros_like(w_log[:,:,:1]), w_log], dim=-2) # (1,H,L+1,K) or (B,H,L+1,S)

        w_log_cum = w_log.cumsum(dim=-2) # (1,H,L,S) or (B,H,L,S)

        # chunked view of w_log
        wc_log = w_log[:,:,1:,:].view(w.size(0),H,N,T,S)
        wc_log_cum = wc_log.cumsum(dim=-2)

        # NOTE - we have to apply the decay weight from TWO ahead.. ONE ahead gets no decay (log==0)
        # pre-applied weights
        # left side is prior chunk (w_inter), right side is current chunk (w_intra)
        # without u...
        # w0   w1   w2   w3   | w4   w5   w6   w7          
        # w1:4 w2:4 w3:4 w4:4 | w4:5 w4:6 w4:7 w4:8
        # with u...
        # w0   w1   w2   w3   | w4   w5   w6   w7          
        # w1:4 w2:4 w3:4 w4:4 | w4:4 w4:5 w4:6 w4:7

        # w_chunk decays the entire current state (representing t-1) to the prior block (t-2)
        w_chunk = wc_log.sum(dim=-2, keepdim=True) # 1HN1S or BHN1S
        # w_inter is the decay to the end of the current block, since it will be applied at the next iteration when current (t) becomes prior (t-1)
        # this formula because e.g. w1:4 = w0:4 - w0:1
        w_inter = w_chunk - wc_log_cum # 1HNTS or BHNTS (w^(T-1) ... w^0)
        # w_intra is the decay from the beginning of the current block (t), since it will be applied to current queries (t) against prior state (representing keys+values up to but not including block t)
        # this formula because e.g. w1:3 = w0:3 - w0
        w_intra = wc_log_cum - wc_log # 1HNTS or BHNTS (w^0 ... w^(T-2))

        w_chunk = list(w_chunk.mT.exp().to(q.dtype).unbind(dim=-3)) # N x 1HS1 or BHS1 !!NOTE THE .mT HERE!!
        w_inter = w_inter.exp().to(q.dtype) # 1HNTS or BHNTS
        w_intra = w_intra.exp().to(q.dtype) # 1HNTS or BHNTS

        # chunked view of r, k, v
        q = q.view(B,H,N,T,K) 
        s = s.view(B,H,N,T,S) 
        k = k.view(B,H,N,T,K) 
        v = v.view(B,H,N,T,V) 
        kv = kv.view(B,H,N,T,K+V) 
        u = u.unsqueeze(2).to(q.dtype) # (1,H,1,1,K)

        # parallel precalculation of chunked (k*wk).mT@v for use in recurrent state calc below
        wskv = (s * w_inter).mT @ kv # BHNS(K+V)
        wskv = list(wskv.unbind(dim=-3)) # N x BHS(K+V)

        # recurrent calculation of all states
        states = []
        for i in range(N):
            states.append(skv_state)
            skv_state = skv_state * w_chunk[i] + wskv[i] # BHS(K+V)
            # equivalent non-precalced version
            #wskv = (s[...,i,:,:] * w_inter[...,i,:,:]).mT @ kv[...,i,:,:]
            #skv_state = skv_state * w_chunk[i] + wskv
        states = torch.stack(states, dim=2) # BHNS(K+V)       

        # parallel calculation of all intra-chunk attention contributions
        wc_log_offset = w_log_cum[:,:,T//2:L:T,None,:] # B,H,N,1,K
        q_decay = (w_log_cum[:,:,:-1,:].view(w.size(0),H,N,T,S) - wc_log_offset).to(precision_dtype).exp() # B,H,N,T,S
        s_inv_decay = (wc_log_offset - w_log_cum[:,:,1:,:].view(w.size(0),H,N,T,S)).to(precision_dtype).exp() # B,H,N,T,S

        # intra-chunk contribution
        qk = (q @ k.mT).tril(-1)                      # TQ @ KT -> TT    (weights by output [query] timeslot and key timeslot)
        mem_attn = (qk @ (s*s_inv_decay)) * q_decay                 # TT @ TS -> TS    (weights by output timeslot and key memoryslot)
        # add u term to mem_attention (NOTE - the tril(-1) above zeroed the diagonal)
        mem_attn = mem_attn + torch.einsum('bhntk,bhntk,bhnts,bhnts->bhnts',q,k,s,u)

        sk_states, sv_states = states.split([K,V], dim=-1) # BHNSK, BHNSV

        # inter-chunk contribution
        mem_attn = mem_attn + (q @ sk_states.mT) * w_intra # TS

        mem_attn = torch.softmax(mem_attn, dim=-1)  # TS -> TS         (softmax over memory slots)

        out = (mem_attn * w_intra) @ sv_states

        # inter-chunk contribution
        seq_attn = ((mem_attn*q_decay) @ (s*s_inv_decay).mT).tril(-1) # TS @ ST -> TT    (keep output timeslot but map key memoryslot back to a key/value timeslot)

        # intra-chunk contribution
        # add u term to seq_attention (NOTE - the tril(-1) above zeroed the diagonal)
        seq_attn = seq_attn + torch.einsum('bhnts,bhnts,bhnts->bhnt',mem_attn,s,u).diag_embed()
        out = out + seq_attn @ v                          # TT @ TV -> TV    (apply key/value timeslot weights to values)
        
        out = out.view(B,H,L,V)
        return out, skv_state

def memtention_recurrent(q_in, s_in, k_in, v_in, w_in, u, skv_state):
    K = k_in.size(-1)
    V = v_in.size(-1)
    L = s_in.size(-2)
    out = []
    sk_state, sv_state = skv_state.split([K,V], dim=-1)
    for t in range(L):
        q, s, k, v, w = q_in[...,t:t+1,:], s_in[...,t:t+1,:], k_in[...,t:t+1,:], v_in[...,t:t+1,:], w_in[...,t:t+1,:]
        sk = s.mT @ k
        sv = s.mT @ v
        out.append( torch.softmax(q @ (sk_state + u.mT * sk).mT, dim=-1) @ (sv_state + u.mT * sv) ) # 1Q @ KS -> 1S, 1S @ SV -> 1V
        sk_state = (w.mT * sk_state) + s.mT @ k # SK
        sv_state = (w.mT * sv_state) + s.mT @ v # SV
    out = torch.cat(out, dim=-2)
    skv_state = torch.cat([sk_state, sv_state], dim=-1)
    return out, skv_state

def memtention_parallel(q, s, k, v, w, u, skv_state):
    # FIXME - use skv_state
    As = w.cumprod(-2)
    Aq = torch.cat([torch.ones_like(w[:,:,:1,:]), w[:,:,:-1,:]], dim=-2).cumprod(-2)
    qk = (q @ k.mT).tril(-1)                      # TQ @ KT -> TT    (weights by output [query] timeslot and key timeslot)
    mem_attn = (qk @ (s/As)) * Aq                 # TT @ TS -> TS    (weights by output timeslot and key memoryslot)
    # add u term to mem_attention (NOTE - the tril(-1) above zeroed the diagonal)
    mem_attn = mem_attn + torch.einsum('bhtk,bhtk,bhts,bhts->bhts',q,k,s,u)
    mem_attn = torch.softmax(mem_attn, dim=-1)  # TS -> TS         (softmax over memory slots)
    seq_attn = ((mem_attn*Aq) @ (s/As).mT).tril(-1) # TS @ ST -> TT    (keep output timeslot but map key memoryslot back to a key/value timeslot)
    # add u term to seq_attention (NOTE - the tril(-1) above zeroed the diagonal)
    seq_attn = seq_attn + torch.einsum('bhts,bhts,bhts->bht',mem_attn,s,u).diag_embed()
    out = seq_attn @ v                          # TT @ TV -> TV    (apply key/value timeslot weights to values)

    # FIXME - calculate skv_state change
    return out, skv_state

def memtention_simple_recurrent(q_in, s_in, k_in, v_in, w_in, skv_state):
    K = k_in.size(-1)
    V = v_in.size(-1)
    L = s_in.size(-2)
    out = []
    sk_state, sv_state = skv_state.split([K,V], dim=-1)
    for t in range(L):
        q, s, k, v, w = q_in[...,t:t+1,:], s_in[...,t:t+1,:], k_in[...,t:t+1,:], v_in[...,t:t+1,:], w_in[...,t:t+1,:]
        sk_state = (w.mT * sk_state) + s.mT @ k # SK
        sv_state = (w.mT * sv_state) + s.mT @ v # SV
        out.append( torch.softmax(q @ sk_state.mT, dim=-1) @ sv_state ) # 1Q @ KS -> 1S, 1S @ SV -> 1V
    out = torch.cat(out, dim=-2)
    skv_state = torch.cat([sk_state, sv_state], dim=-1)
    return out, skv_state

def memtention_simple_parallel(q, s, k, v, w, skv_state):
    # FIXME - use skv_state
    A = w.cumprod(-2)
    qk = (q @ k.mT).tril()                      # TQ @ KT -> TT    (weights by output [query] timeslot and key timeslot)
    mem_attn = (qk @ (s/A)) * A                 # TT @ TS -> TS    (weights by output timeslot and key memoryslot)
    mem_attn = torch.softmax(mem_attn, dim=-1)  # TS -> TS         (softmax over memory slots)
    seq_attn = ((mem_attn*A) @ (s/A).mT).tril() # TS @ ST -> TT    (keep output timeslot but map key memoryslot back to a key/value timeslot)
    out = seq_attn @ v                          # TT @ TV -> TV    (apply key/value timeslot weights to values)

    # FIXME - calculate skv_state change
    return out, skv_state

def sanity_check():
    B = 1
    H = 1
    T = 8
    S,K,V = 5,3,2
    q = torch.rand(B,H,T,K)
    s = torch.rand(B,H,T,S)
    k = torch.rand(B,H,T,K)
    v = torch.rand(B,H,T,V)
    w = torch.rand(B,H,T,S)
    u = torch.rand(B,H,1,S)
    skv_state = torch.zeros(B,H,S,K+V)

    # recurrent
    out, _ = memtention_recurrent(q,s,k,v,w,u,skv_state)
    print(out)

    # parallel
    out, _ = memtention_parallel(q,s,k,v,w,u,skv_state)
    print(out)

    # chunked
    out, _ = memtention_inner(q,s,k,v,w,u,skv_state,chunk_len=4)
    print(out)

if __name__ == "__main__":
    sanity_check()
    exit()


class QWiKSilver_AttentionSubLayer(model.core.TransformerLayerPart, model.interface.IAttentionSubLayer):
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
            self.tm_w1 = nn.Parameter(torch.empty(args.n_embd, TIME_MIX_EXTRA_DIM * 6).uniform_(-0.01, 0.01))
            self.tm_w2 = nn.Parameter(torch.zeros(6, TIME_MIX_EXTRA_DIM, args.n_embd))
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
        s_ratio = 1.0
        S = int(K * s_ratio)

        B, T, C = x.size()

        xx = x
        shifted_x = self.time_shift(x) - xx
        #shifted_x = torch.cat((shifted_x.unsqueeze(0), xx[:-1,:])) - xx

        xxx = xx + shifted_x * self.x_maa
        xxx = torch.tanh(xxx @ self.tm_w1).view(B*T, 6, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.tm_w2).view(6, B, T, -1)
        mw, ms, mk, mv, mr, mg = xxx.unbind(dim=0)

        wx = xx + shifted_x * (self.w_maa + mw)
        sx = xx + shifted_x * (self.k_maa + ms)
        kx = xx + shifted_x * (self.k_maa + mk)
        vx = xx + shifted_x * (self.v_maa + mv)
        rx = xx + shifted_x * (self.r_maa + mr)
        gx = xx + shifted_x * (self.g_maa + mg)

        q = self.receptance(rx).view(B, T, H, K).transpose(1, 2) # BHTK
        s = self.key(sx).view(B, T, KVH, S).transpose(1, 2)      # BHTS
        k = self.key(kx).view(B, T, KVH, K).transpose(1, 2)      # BHTK
        v = self.value(vx).view(B, T, KVH, V).transpose(1, 2)    # BHTV
        g = self.gate(gx)

        # rotate queries and keys via RoPE / XPos
        q, k = self.rotary_positional_embedding((q, k))

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

        skv_state = recurrent_memory
        if skv_state is None:
            skv_state = torch.zeros(B, H, S, K+V, device=q.device, dtype=q.dtype)

        if skv_state.dtype != q.dtype:
            skv_state = skv_state.contiguous().to(q.dtype)

        w = time_decay.view(1, H, 1, K)
        w = w + (torch.tanh(wx @ self.td_w1) @ self.td_w2).view(B, T, H, K).transpose(1, 2) # BHTK
        w = torch.exp(-torch.exp(w))

        u = time_first.view(1, H, 1, K)

        out, kv_state = memtention_inner(q, s, k, v, w, u, skv_state)

        # TransNormer style normalization after multiplying qkv (same as separate groupnorm of each head but w/o a scaling parameter)
        # normalize each head
        out = norm.Norm.F(out)

        # squeeze all the head embeddings together into a single dimension
        out = out.transpose(1, 2).reshape(B, T, H*V) # (B, H, T, V) -> (B, T, H*V)

        out = self.output(out * g)

        return out
    