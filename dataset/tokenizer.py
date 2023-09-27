import numpy as np
import torch
from functools import partial
import typing

import cfgctx

def tokenize(input, tokenizer):
    temp_max_length = getattr(tokenizer, 'model_max_length', None)
    tokenizer.model_max_length=int(1e30) # to avoid warnings when tokenizing long strings
    output = torch.tensor(tokenizer(input['text'])['input_ids'])
    tokenizer.model_max_length = temp_max_length
    text = input['text']
    return output

def tokenize_join_and_slice(input_batch : list[dict], tokenizer, block_size):
    # temporarily set tokenizer.model_max_length to avoid warnings when tokenizing long strings
    temp_max_length = getattr(tokenizer, 'model_max_length', None)
    tokenizer.model_max_length=int(1e30)

    # join the text strings from each input in the input_batch together via the eos_token
    text = str(tokenizer.eos_token).join([entry['text'] for entry in input_batch])

    # tokenize the result
    toks = torch.tensor(tokenizer(text)['input_ids'])

    # split the result into a new output batch of token chunks of block_size_plus length
    block_size_plus = block_size + 1
    output_batch = [toks[i*block_size_plus:(i+1)*block_size_plus] for i in range(len(toks)//block_size_plus)]

    # reset tokenizer.model_max_length
    tokenizer.model_max_length = temp_max_length

    return output_batch # different size than input_batch

    #return dict(input_ids=input_ids)

def tokenize_join_and_slice_in_context(input_batch : list[dict]):
    if input_batch is None:
        return None
    return tokenize_join_and_slice(input_batch, cfgctx.tokenizer, cfgctx.block_size)

class Callable():
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    def forward(self):
        raise NotImplementedError()

class TokenizeMergeAndSplit(Callable):
    def __init__(self, tokenizer, block_size : int):
        self.tokenizer = tokenizer
        self.block_size = block_size

    def forward(self, dataset):
        return dataset.map(lambda x: tokenize_merge_and_split(x, self.tokenizer, self.block_size))#, batched=True, remove_columns=dataset.column_names)

# def split_max_tokens_fn(data, tokenizer, max_tokens):
#     temp_max_length = getattr(tokenizer, 'model_max_length', None)
#     tokenizer.model_max_length=int(1e30) # to avoid warnings when tokenizing long strings
#     toks = tokenizer(data['text'])['input_ids']
#     return [toks[i*max_tokens:(i+1)*max_tokens] for i in range(len(toks) // max_tokens)]
