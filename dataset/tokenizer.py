import torch

def tokenize(input, tokenizer):
    temp_max_length = getattr(tokenizer, 'model_max_length', None)
    tokenizer.model_max_length=int(1e30) # to avoid warnings when tokenizing long strings
    output = torch.tensor(tokenizer(input['text'])['input_ids'])
    tokenizer.model_max_length = temp_max_length
    return output

def tokenize_join_and_slice_input_ids(data, tokenizer, block_size):
    # temporarily set tokenizer.model_max_length to avoid warnings when tokenizing long strings
    temp_max_length = getattr(tokenizer, 'model_max_length', None)
    tokenizer.model_max_length=int(1e30)

    # join the text strings from each input in the input_batch together via the eos_token
    text = str(tokenizer.eos_token).join(data['text'])

    # tokenize the result
    toks = torch.tensor(tokenizer(text)['input_ids'])

    # split the result into a new output batch of token chunks of block_size_plus length
    block_size_plus = block_size + 1
    output_batch = [toks[i*block_size_plus:(i+1)*block_size_plus] for i in range(len(toks)//block_size_plus)]

    # reset tokenizer.model_max_length
    tokenizer.model_max_length = temp_max_length

    return dict(input_ids=output_batch) # different size than input_batch

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

def tokenize_crop_join_and_slice_input_ids(data, tokenizer, block_size : int, crop_n_blocks : int):
    # temporarily set tokenizer.model_max_length to avoid warnings when tokenizing long strings
    temp_max_length = getattr(tokenizer, 'model_max_length', None)
    tokenizer.model_max_length=int(1e30)

    bs1 = block_size+1
    tokenized = tokenizer(data['text'])['input_ids']
    tokenized = map(lambda x: x[:bs1*crop_n_blocks], tokenized)

    # join the text strings from each input in the input_batch together via the eos_token

    dt = torch.long
    text_tensors = [torch.tensor(tokens, dtype=dt) for tokens in tokenized]
    eos_tensor = torch.tensor([tokenizer.eos_token_id], dtype=dt)
    output_batch = []
    i = -1
    while i < len(text_tensors) - 1:
        i += 1
        text_tensor = text_tensors[i]
        if text_tensor.size(0) >= bs1:
            output_batch = output_batch + [text_tensor[i*bs1:(i+1)*bs1] for i in range(len(text_tensor)//bs1)]
        else:
            while i < len(text_tensors) - 1:
                i += 1
                text_tensor = torch.cat([text_tensor, eos_tensor, text_tensors[i]])
                if text_tensor.size(0) >= bs1:
                    output_batch.append(text_tensor[:bs1])
                    break

    # reset tokenizer.model_max_length
    tokenizer.model_max_length = temp_max_length

    return dict(input_ids=output_batch) # different size than input_batch

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
        return dataset.map(lambda x: tokenize_join_and_slice_input_ids(x, self.tokenizer, self.block_size))#, batched=True, remove_columns=dataset.column_names)

