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
    # pk = (w * pk) + p.mT @ k    # PK
    # pv = (w * pv) + p.mT @ v    # PV
    # out = torch.softmax(q @ pk.mT, dim=-1) @ pv # 1Q @ KP -> 1P, 1P @ PV -> 1V

    # non-decayed simplified recurrence
    # pk = pk + p.mT @ k    # PK
    # pv = pv + p.mT @ v    # PV
    # out = torch.softmax(q @ pk.mT, dim=-1) @ pv # 1Q @ KP -> 1P, 1P @ PV -> 1V

    # non-decayed parallel
    # torch.softmax(q @ torch.cumsum(p.mT @ k, dim=-3).mT, dim=-1) @ torch.cumsum(p.mT @ v, dim=-3) # T1Q @ TKP -> T1P, T1P @ TPV -> T1V
    #=torch.softmax(q @ torch.cumsum(k.mT @ p, dim=-3), dim=-1) @ torch.cumsum(p.mT @ v, dim=-3) # T1Q @ TKP -> T1P, T1P @ TPV -> T1V
    #=torch.softmax((q @ k.mT).tril() @ p, dim=-2).unsqueeze(-2) @ torch.cumsum(p.mT @ v, dim=-3) # TT @ TP -> TP, T1P @ TPV -> T1V
    #=(torch.softmax((q @ k.mT).tril() @ p, dim=-2) @ p.mT).tril() @ v # TT @ TP -> TP, TP @ PT @ TV -> TV

    # 1. o = q @ k.mT          # TQ @ KT -> TT    (weights by output [query] timeslot and key timeslot)
    # 2. o = o.tril() @ p      # TT @ TP -> TP    (weights by output timeslot and key memoryslot)
    # 3. o = softmax(o) @ p.mT # TP @ PT -> TT    (keep output timeslot but map key memoryslot back to a key/value timeslot)
    # 4. o = o.tril() @ v      # TT @ TV -> TV    (apply key/value timeslot weights to values)    

    # this is interesting in terms of interpretation:
    # steps 2 and 3 essentially translate from timeslot to memoryslot and then back to timeslot 'undoing' the operation
    # but importantly applying softmax in the middle, to give us full traditional attention fidelity!
    # its like 
    # @= p    # do
    # @= p.mT # undo
    # if you eliminate steps 2 and 3 you get linear attention!

    def parallel(p,q,k,v,w): # all (B,H,T,D)
        A = w.cumprod(0)

        qk = (q @ k.mT).tril()                      # TQ @ KT -> TT    (weights by output [query] timeslot and key timeslot)
        mem_attn = (qk @ (p/A)) * A                 # TT @ TP -> TP    (weights by output timeslot and key memoryslot)
        mem_attn = torch.softmax(mem_attn, dim=-1)  # TP -> TP         (softmax over memory slots)
        seq_attn = ((mem_attn*A) @ (p/A).mT).tril() # TP @ PT -> TT    (keep output timeslot but map key memoryslot back to a key/value timeslot)
        return seq_attn @ v                         # TT @ TV -> TV    (apply key/value timeslot weights to values)
    
def sanity_check():
    T = 2
    P,K,V = 3,3,3
    w = torch.rand(T,P)
    q = torch.rand(T,K)
    k = torch.rand(T,K)
    p = torch.rand(T,P)
    v = torch.rand(T,V)
    pk = torch.zeros(P,K)
    pv = torch.zeros(P,V)

    # recurrent
    out = []
    for t in range(T):
        pk = (w[t:t+1].mT * pk) + p[t:t+1].mT @ k[t:t+1]
        pv = (w[t:t+1].mT * pv) + p[t:t+1].mT @ v[t:t+1]
        out.append( torch.softmax(q[t:t+1] @ pk.mT, dim=-1) @ pv )
    out = torch.cat(out, dim=0)
    print(out)

    # parallel
    A = w.cumprod(0)
    qk = (q @ k.mT).tril()                      # TQ @ KT -> TT    (weights by output [query] timeslot and key timeslot)
    mem_attn = (qk @ (p/A)) * A                 # TT @ TP -> TP    (weights by output timeslot and key memoryslot)
    mem_attn = torch.softmax(mem_attn, dim=-1)  # TP -> TP         (softmax over memory slots)
    seq_attn = ((mem_attn*A) @ (p/A).mT).tril() # TP @ PT -> TT    (keep output timeslot but map key memoryslot back to a key/value timeslot)
    out = seq_attn @ v                         # TT @ TV -> TV    (apply key/value timeslot weights to values)
    print(out)

sanity_check()
exit()
