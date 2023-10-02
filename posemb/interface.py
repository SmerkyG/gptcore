from typing import Any, Optional, Tuple, List, Iterable

import torch
import torch.nn as nn
from torch import Tensor

class IPositionalEmbedding():
    def forward(self, x : Tuple[Tensor, Tensor]): pass

class IQueryKeyEmbedding():
    def forward(self, x : Tuple[Tensor, Tensor]): pass

