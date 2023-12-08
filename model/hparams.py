import torch.nn as nn

from dataclasses import dataclass, field

from functools import partial

from posemb.interface import IPositionalEmbedding, IQueryKeyEmbedding
from model.interface import IFeedForwardSubLayer, IAttentionSubLayer

from typing import Callable, Any

from util.config import Factory

def field_default(fn):
    return field(default_factory=fn)

@dataclass
class HParams():
    # FIXME - can we pick up vocab size from tokenizer/dataset somehow?
    vocab_size: int         # number of possible tokens in the tokenizer's vocabulary, ideally rounded up to 8 for efficiency
    max_sequence_length : int   # maximum possible sequence length this model can process in a single call to forward() - this is important for precalculating masks and rotary embeddings

    n_layer : int = 12      # number of layers in this transformer
    d_model : int = 768     # dimension of main model embedding space
    n_head : int = 12       # number of heads (base_head_size = int(d_model / n_head))

    n_kv_head_ratio : float = 1 # ratio of kv heads to n_head for grouped-query attention (GQA)

    d_qk_ratio : float = 1  # ratio of query_head_size and key_head_size to base_head_size
    d_v_ratio : float = 1   # ratio of value_head_size to base_head_size

    feedforward_d_model_ratio : float = 4   # ratio of feedforward network hidden embedding size to d_model
   
    dropout : float = 0.0   # dropout is more useful for finetuning than pretraining

    train_hardened : bool = False   # train FFF layers with hardened boundaries
    n_leaf : int = 32               # leaf count, must be power of 2

    rotary_positional_embedding_factory : Callable[..., IQueryKeyEmbedding | nn.Identity] = field_default(lambda: Factory(nn.Identity))
