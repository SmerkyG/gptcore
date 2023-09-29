from typing import Any, Optional, Tuple, List, Iterable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import model
import model.core
import sampler

from util.config import Factory
import cli
import cfgctx
import torch.amp
from contextlib import nullcontext
from dataclasses import dataclass, field

def field_default(fn):
    return field(default_factory=fn)

@dataclass
class Evaluator(cli.IEvaluator):
    sampler_factory:Factory=field_default(lambda: Factory(sampler.TopKPTailFreeSampler, temperature=1.0, top_p=0.7))

    def eval(self, cfg : cli.ConfigBase):
        max_new_tokens = 500
        device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
        dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
        device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

        out_dir = 'out'
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        hparams = checkpoint['hparams']
        checkpoint = torch.load(ckpt_path, map_location=device)
        model = model.core.Decoder(hparams)
        state_dict = checkpoint['model']
        unwanted_prefix = 'model._orig_mod.'
        wanted_prefix = ''
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[wanted_prefix + k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        gen = Generator(model)

        starter_text = "<|endoftext|>In a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."
        #starter_text = starter_text + starter_text
        tokenized_starter_text = cfgctx.tokenizer(starter_text)['input_ids']
        starter_ids = tokenized_starter_text[-1025:-1]
        predicted = tokenized_starter_text[-1024:]
        x = (torch.tensor(starter_ids, dtype=torch.long, device=device)[None, ...])
        y = (torch.tensor(predicted, dtype=torch.long, device=device)[None, ...])

        # with torch.no_grad():
        #     logits = model.forward(x)
        #     predicted_labels = logits.argmax(dim=-1)
        #     acc = predicted_labels.eq(y).sum() / float(y.size(0)*y.size(1))
        #     print(f"acc {float(acc)}")

        #     print(tokenizer.decode(predicted_labels.squeeze(0).tolist()))
        

        sampler = self.sampler_factory()
        print(cfgctx.tokenizer.decode(starter_ids))
        print("...")
        with torch.no_grad():
            with ctx:
                for tok in gen.generate_tokens(x, max_new_tokens, sampler, alpha_frequency = 0.25, alpha_presence = 0.25, alpha_decay = 1.0 / 200):
                    print(cfgctx.tokenizer.decode(tok.item()), end='')
                    #print(decode(y[0].tolist()))
                print('')
                print('---------------')

class Generator(nn.Module):
    def __init__(self, model : model.core.IEncoderDecoder):
        super().__init__()
        self.model = model
        self.hparams = self.model.hparams

    # call this first, if using encoder-decoder combo
    def encode(self, x : Tensor):
        return self.model.encode(x, None, None)
    
    def decode(self, x : Tensor, encoder_output : Tensor = None, recurrent_memory : Optional[list[Tensor]] = None):
        return self.model.decode(x, encoder_output, recurrent_memory)

    def forward(self, x : Tensor, encoder_output : Tensor = None, recurrent_memory : Optional[list[Tensor]] = None):
        return self.model.forward(x, encoder_output, recurrent_memory)

    def next_token(self, input_tokens : Tensor, sampler = sampler.TopKPTailFreeSampler()):
        x = self.decode(input_tokens if input_tokens.size(1) <= self.hparams.block_size else input_tokens[..., -self.hparams.block_size:])
        next_token = sampler(x[..., -1, :])
        return next_token, torch.cat((input_tokens, next_token), dim=-1)

    def generate_tokens_simple(self, input_tokens : Tensor, num_outputs : int, sampler = sampler.TopKPTailFreeSampler()):
        output_tokens = input_tokens
        for _ in range(num_outputs):
            next_token, output_tokens = self.next_token(output_tokens, sampler)
            yield next_token

    def generate_tokens(self, input_tokens : Tensor, num_outputs : int, sampler = sampler.TopKPTailFreeSampler(), alpha_frequency :float = 0, alpha_presence : float = 0, alpha_decay : float = 0):
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

    def create_recurrent_memory(self, input_tokens : Optional[Tensor], sampler = sampler.TopKPTailFreeSampler()) -> Tuple[int, Tensor]:
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
    
    def next_token_recurrent(self, latest_x : Tensor, recurrent_memory : Tensor, sampler = sampler.TopKPTailFreeSampler()):
        return sampler(self.decode(latest_x, recurrent_memory))
    
    def generate_tokens_recurrent(self, next_token : int, recurrent_memory : Tensor, num_outputs, sampler = sampler.TopKPTailFreeSampler()):
        for _ in range(num_outputs):
            next_token = self.next_recurrent(next_token, recurrent_memory, sampler)
            yield next_token

    def generate_tokens_recurrent_from_scratch(self, num_outputs : int, sampler = sampler.TopKPTailFreeSampler()):
        next_token, recurrent_memory = self.create_recurrent_memory(input_tokens = None)
        return self.generate_tokens_recurrent(next_token, recurrent_memory, num_outputs, sampler)

