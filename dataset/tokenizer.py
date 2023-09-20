import numpy as np
import torch
from functools import partial

def tokenize_merge_and_split(tokenizer, block_size, data):
    temp_max_length = getattr(tokenizer, 'model_max_length', None)
    tokenizer.model_max_length=int(1e30) # to avoid warnings when tokenizing long strings
    text = str(tokenizer.eos_token).join(data['text'])
    toks = torch.tensor(tokenizer(text)['input_ids'])
    block_size_plus = block_size + 1
    input_ids = [toks[i*block_size_plus:(i+1)*block_size_plus] for i in range(len(toks)//block_size_plus)]
    tokenizer.model_max_length = temp_max_length
    return dict(input_ids=input_ids)

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
        return dataset.map(partial(tokenize_merge_and_split, self.tokenizer, self.block_size), batched=True, remove_columns=dataset.column_names)

