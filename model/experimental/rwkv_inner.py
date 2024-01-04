import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

# 32 is optimal chunk length (longer will use too much memory, shorter is inefficient)
def rwkv_inner(s,r,k,v,w,u,chunk_len=32):
    """
    expects
    s : (B,H,K,V) # recurrent kv state
    r : (B,H,L,K)
    k : (B,H,L,K)
    v : (B,H,L,V)
    w : (B,L,H,K) or (1,L,H,K)
    u : (1,H,K)
    """
    B,H,L,K = k.size()
    V = v.size(-1)
    T = chunk_len

    if L == 1:
        kv = k @ v
        out = r @ (s + u * kv)
        s = w * s + kv
        return out, s
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
        w_log = w.transpose(-3,-2).float().log() # (1,H,L,K) or (B,H,L,K)

        w_log_cum = w_log.cumsum(dim=-2) # (1,H,L,K) or (B,H,L,K)
        w_log_cum = torch.cat([w_log_cum, w_log_cum[:,:,-1:]], dim=-2) # (1,H,L+1,K) or (B,H,L+1,K)

        # chunked view of w_log
        wc_log = w_log.view(w.size(0),H,N,T,K)
        # ws decays the entire current state (representing t-1) to the prior block (t-2)
        ws = wc_log.sum(dim=-2, keepdim=True) # 1HN1K or BHN1K
        # wk is the decay to the end of the current block, since it will be applied at the next iteration when current (t) becomes prior (t-1)
        wk = ws - wc_log.cumsum(dim=-2) - wc_log # 1HNTK or BHNTK (w^(T-2) ... w^-1)
        # w_intra is the decay from the beginning of the current block (t), since it will be applied to current queries (t) against prior state (representing keys+values up to but not including block t)
        w_intra = wc_log.cumsum(dim=-2) # 1HNTK or BHNTK (w^0 ... w^(T-1))

        ws = list(ws.exp().to(r.dtype).unbind(dim=-3)) # N x 1H1K or BH1K
        wk = wk.exp().to(r.dtype) # 1HNTK or BHNTK
        w_intra = w_intra.exp().to(r.dtype) # 1HNTK or BHNTK

        u = u.transpose(0,1).to(r.dtype) # (H,1,K)

        # chunked view of r, k, v
        r = r.view(B,H,N,T,K) 
        k = k.view(B,H,N,T,K) 
        v = v.view(B,H,N,T,V)

        # parallel calculation of all intra-chunk attention contributions
        wc_log_offset = w_log_cum[:,:,T//2:L:T,None,:] # B,H,N,1,K
        r_decay = (w_log_cum[:,:,:-1,:].view(w.size(0),H,N,T,K) - wc_log_offset).to(precision_dtype).exp() # B,H,N,T,K
        k_inv_decay = (wc_log_offset - w_log_cum[:,:,1:,:].view(w.size(0),H,N,T,K)).to(precision_dtype).exp() # B,H,N,T,K
        a = ((r*r_decay) @ (k*k_inv_decay).mT).to(r.dtype).tril(-1) # B,H,N,T,T
        # add u term to attention (NOTE - the tril(-1) above zeroed the diagonal)
        a = a + torch.einsum('bhntk,bhntk->bhnt', r, u.unsqueeze(-2) * k).diag_embed()
        out = a @ v # BHNTV
        # alternate way of adding in u
        #outc = outc + torch.einsum('bhntk,bhntk,bhntv->bhntv', rc, u.unsqueeze(-2) * kc, vc) 

        # parallel precalculation of chunked (k*wk).mT@v for use in recurrent state calc below
        wkv = (k * wk).mT @ v # BHNKV
        wkv = list(wkv.unbind(dim=-3)) # N x BHKV

        # recurrent calculation of all states
        states = []
        for i in range(N):
            states.append(s)
            s = s * ws[i] + wkv[i] # BHKV
        states = torch.stack(states, dim=2) # BHNKV       

        # parallel application of all r to states
        out = out + (r * w_intra) @ states # BHNTV
        out = out.view(B,H,L,V).transpose(1,2)
        return out, s
            