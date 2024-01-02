import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

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

def memtention_inner(s,q,p,kv,w,u):
    """
    expects
    s : (B,H,P,K+V) # recurrent pkv state
    q : (B,L,H,Q) # query
    p : (B,L,H,P) # put
    kv : (B,L,H,K+V) # key and value side by side
    w : (L,H,P) # decay
    u : (L,H,P) # bonus
    """
    B,H,L,K = q.size()
    Q = K
    V = kv.size(-1) - K
    P = K # FIXME - make this resizable to allow for varying sized memory bank
    T = 32 # optimal chunk length (longer will use too much memory, shorter is inefficient)


    # simplified recurrence
    # pks = w * ks + p.mT @ k # PK
    # pvs = w * vs + p.mT @ v # PV
    # out = torch.softmax(q @ pks.mT, dim=-1) @ pvs # 1Q @ KP -> 1P, 1P @ PV -> 1V

    if L == 1:
        pkv = p @ kv
        ram_pk, ram_pv = (s + u * pkv).split()
        # attention 1Q @ KP -> 1P, 1P @ PV -> 1V
        out = torch.softmax(q @ ram_pk.mT, dim=-1) @ ram_pv
        s = w * s + pkv
        return out, s
    else:
        # FIXME - implement parallel and recurrent chunked version

        assert False
            