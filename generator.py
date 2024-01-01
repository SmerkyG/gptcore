from typing import Any, Optional, Tuple, List, Iterable
import typing
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import model
from model.core import IEncoderDecoder
import sampler

class Generator(nn.Module):
    def __init__(self, model : IEncoderDecoder, sampler : nn.Module = sampler.RepetitionPenalizer(sampler.TopKPTailFreeSampler())):
        super().__init__()

        # FIXME - unwrap model if compiled
        #assert(isinstance(model, IEncoderDecoder))
        
        self.model = model
        self.sampler = sampler
        # FIXME - ensure hparams exists typesafely somehow
        self.hparams = self.model.layers[0].hparams

        self.decoder_initialized = False
        self.clear_encoder_state()
        self.clear_decoder_state()

    # call this first, if using encoder-decoder combo
    def encode(self, x : Tensor):
        self.encoder_output = self.model.encode(x, None, None)

    def ingest(self, x : Tensor):
        # init decoder state if needed
        if self.tokens is None:
            self.recurrent_memory = None # FIXME - call self.model.create_recurrent_memory()
            self.tokens = torch.tensor([[]], dtype=x.dtype, device=x.device)

        if self.recurrent_memory is None and self.tokens.size(-1) + x.size(-1) > self.hparams.max_sequence_length:
            # non-recurrent, can only use up to max context length
            raise Exception("Context length exceeded")
        self.tokens = torch.cat([self.tokens, x[..., :-1]], dim=-1)
        _ = self.model.decode(self.tokens, self.encoder_output, self.recurrent_memory)
        next_token_tensor = x[..., -1:]
        if self.recurrent_memory is None:
            self.tokens = torch.cat([self.tokens, next_token_tensor], dim=-1)
        else:
            self.tokens = next_token_tensor

    def predict(self, num_outputs : int) -> typing.Generator[Tensor, None, None]:
        if self.finish_reason == 'length':
            raise Exception("Context length exceeded")
        if self.tokens is None:
            raise Exception("Must call ingest() before calling decode(), even if only to ingest a single BOS/EOS token")

        self.finish_reason = None
        for _ in range(num_outputs):
            logits = self.model.decode(self.tokens, self.encoder_output, self.recurrent_memory)
            next_token_tensor = self.sampler(logits)
            #next_token_tensor = torch.tensor([[next_token_id]], dtype=self.tokens.dtype, device=self.tokens.device)
            if self.recurrent_memory is None:
                self.tokens = torch.cat([self.tokens, next_token_tensor], dim=-1)
            else:
                self.tokens = next_token_tensor
            yield next_token_tensor
            if self.tokens.size(-1) > self.hparams.max_sequence_length:
                # in non-recurrent mode, can only use up to max context length, so stop the generator here
                self.finish_reason = "length"
                self.clear_decoder_state()
                return
        self.finish_reason = "stop"
            
    def get_finish_reason(self):
        return self.finish_reason

    def clear_encoder_state(self):
        self.encoder_output = None

    def clear_decoder_state(self):
        self.recurrent_memory = None
        self.tokens = None
        self.finish_reason = None


