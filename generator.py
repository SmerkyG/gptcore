from typing import Any, Optional, Tuple, List, Iterable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import picogpt
import sampler

class Generator(nn.Module):
    def __init__(self, model : core.IEncoderDecoder):
        super().__init__()
        self.model = model
        self.hparams = self.model.hparams

    # call this first, if using encoder-decoder combo
    def encode(self, x : Tensor):
        return self.encoder(x, None, None)
    
    def decode(self, x : Tensor, encoder_output : Tensor = None, recurrent_memory : Optional[list[Tensor]] = None):
        return self.model.decode(x, encoder_output, recurrent_memory)

    def forward(self, x : Tensor, encoder_output : Tensor = None, recurrent_memory : Optional[list[Tensor]] = None):
        return self.model.forward(x, encoder_output, recurrent_memory)

    def next_token(self, input_tokens : Tensor, sampler = sampler.TopKSampler()):
        x = self.decode(input_tokens if input_tokens.size(1) <= self.hparams.block_size else input_tokens[..., -self.hparams.block_size:])
        next_token = sampler(x[..., -1, :])
        return next_token, torch.cat((input_tokens, next_token), dim=-1)

    def generate_tokens_simple(self, input_tokens : Tensor, num_outputs : int, sampler = sampler.TopKSampler(), alpha_frequency :float = 0, alpha_presence : float = 0, alpha_decay : float = 0):
        output_tokens = input_tokens
        for _ in range(num_outputs):
            next_token, output_tokens = self.next_token(output_tokens, sampler)
            yield next_token

    def generate_tokens(self, input_tokens : Tensor, num_outputs : int, sampler = sampler.TopKSampler(), alpha_frequency :float = 0, alpha_presence : float = 0, alpha_decay : float = 0):
        output_tokens = input_tokens
        token_map = dict()
        for _ in range(num_outputs):
            logits = self.decode(output_tokens if output_tokens.size(1) <= self.hparams.block_size else output_tokens[..., -self.hparams.block_size:])
            for token in token_map:
                logits[0, -1, token] -= alpha_presence + token_map[token] * alpha_frequency
                #logits[0, -1, token] *= (1.0 - alpha_presence) ** token_map[token] #alpha_presence + token_map[token] * alpha_frequency
                token_map[token] *= 1.0 - alpha_decay
            next_token = sampler(logits[0, -1, :]).unsqueeze(dim=0)
            output_tokens = torch.cat((output_tokens, next_token), dim=-1)
            token_map[next_token] = token_map[next_token] + 1 if next_token in token_map else 1
            yield next_token

    def create_recurrent_memory(self, input_tokens : Optional[Tensor], sampler = sampler.TopKSampler()) -> Tuple[int, Tensor]:
        # a recurrent state for each layer, each a single (Q*H, Q*H) tensor representing all the state of all the heads for that layer
        Q = int(self.hparams.d_qk_ratio * self.hparams.d_model / self.hparams.n_head)
        recurrent_memory = [torch.zeros(Q * self.hparams.n_head, Q * self.hparams.n_head).unsqueeze(0).repeat(input_tokens.shape[0], 1, 1) for _ in range(self.hparams.n_layer)]
        if input_tokens is None or input_tokens.size(-1) == 0:
            # FIXME - should this be a special starting token for unconditional generation?
            token = 50297
            x = self.decode(token, recurrent_memory)
        else:
            for token in input_tokens:
                x = self.decode(token, recurrent_memory)
            next_token = sampler(x)
        return next_token, recurrent_memory
    
    def next_token_recurrent(self, latest_x : Tensor, recurrent_memory : Tensor, sampler = sampler.TopKSampler()):
        return sampler(self.decode(latest_x, recurrent_memory))
    
    def generate_tokens_recurrent(self, next_token : int, recurrent_memory : Tensor, num_outputs, sampler = sampler.TopKSampler()):
        for _ in range(num_outputs):
            next_token = self.next_recurrent(next_token, recurrent_memory, sampler)
            yield next_token

    def generate_tokens_recurrent_from_scratch(self, num_outputs : int, sampler = sampler.TopKSampler()):
        next_token, recurrent_memory = self.create_recurrent_memory(input_tokens = None)
        return self.generate_tokens_recurrent(next_token, recurrent_memory, num_outputs, sampler)

