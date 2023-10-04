from typing import Any, Optional, Tuple, List, Iterable

import torch
import torch.nn as nn
from torch import Tensor

from model.interface import IModule 

class IPositionalEmbedding(IModule):
    def forward(self, x : Tensor): pass

class IQueryKeyEmbedding(IModule):
    def forward(self, x : Tuple[Tensor, Tensor]): pass

