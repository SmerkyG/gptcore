from util.config import Factory

from typing import Any, Optional, Tuple, List, Iterable
from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dataclasses import dataclass

from dataclasses import dataclass, field

from functools import partial

from posemb.interface import IPositionalEmbedding, IQueryKeyEmbedding
from model.interface import IFeedForwardSubLayer, IAttentionSubLayer

from typing import Callable, Any

def factory(*args, **kwargs):
    return field(default_factory=partial(Factory, *args, **kwargs))

@dataclass
class HParams():
    # FIXME - can we pick up vocab size from tokenizer/dataset?
    vocab_size: int = 50304
    block_size : int = 0 # this gets filled in from Ctx # FIXME - what if this is different for encoder and decoder?

    n_layer : int = 12
    n_head : int = 12
    d_model : int = 768

    # for RetNet we need these parameters to be different so that the sizes match:
    #  d_v_ratio must be 2 for RetNet instead of 1 for GPT
    #  feedforward_d_model_ratio must be 2 for RetNet instead of 4 for GPT
    #  and rotary, not xpos positional embedding is used
    d_qk_ratio : float = 1
    d_v_ratio : float = 1 # FIXME - the reason it's nice for these to be global settings is that they are often important to stay the same when switching attention modules

    # dropout is more useful for finetuning than pretraining
    dropout : float = 0.0

    positional_embedding_factory : Callable[..., IPositionalEmbedding] = factory('model.core.NoOpModule')
    rotary_positional_embedding_factory : Callable[..., IQueryKeyEmbedding] = factory('model.core.NoOpModule')

    self_attention_sublayer_factory : Callable[..., IAttentionSubLayer] = factory('model.core.AttentionSubLayer')

    cross_attention_sublayer_factory : Callable[..., IAttentionSubLayer] = factory('model.core.NoOpModule')

    feedforward_d_model_ratio : float = 4
    feedforward_sublayer_factory : Callable[..., IFeedForwardSubLayer] = factory('model.core.FFN')
