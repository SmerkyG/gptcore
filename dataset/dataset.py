import numpy as np
import os
import torch

class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, vocab_size=50304, block_size=1024):
        self.vocab_size = vocab_size
        self.block_size = block_size
    def __len__(self):
        return 1024*1024
    def __getitem__(self, idx: int):
        d = torch.randint(self.vocab_size, (self.block_size+1,))
        return dict(input_ids=d)

class MMapDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir : str, data_filename : str, block_size=256, mask_size=1):
        data_path = os.path.join(data_dir, data_filename)
        data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.data_len = len(data)
       
        self.data = data
        self.block_size = block_size
        self.mask_size = mask_size
        #self.n_blocks = (self.data.size - 1) // self.block_size # -1 is to allow for one extra token at the end!
        # FIXME - not as correct, but we are trying to simulate an old run!
        self.n_blocks = self.data.size // self.block_size - 1 # -1 is to allow for one extra token at the end!

    def __len__(self):
        return self.n_blocks

    def __getitem__(self, idx: int):
        d = self.data[idx*self.block_size:self.mask_size+(idx+1)*self.block_size].astype(np.int64)
        return torch.from_numpy(d[:-self.mask_size]), torch.from_numpy(d[self.mask_size:])

import datasets

class HFDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, hf_dataset : datasets.Dataset):
        self.hf_dataset = hf_dataset

    def __iter__(self):
        for d in self.hf_dataset:
            yield d['input_ids'].squeeze(0)[:-1], d['input_ids'].squeeze(0)[1:]

    #def __len__(self):
    #    return self.hf_dataset

    # def __getitem__(self, idx: int):
    #     d = self.hf_dataset[idx]
    #     return d[:-1], d[1:]

    
class HFDataLoaderWrapper(torch.utils.data.DataLoader):
    def __init__(self, hf_dataloader : torch.utils.data.DataLoader):
        super().__init__()
        self.hf_dataloader = hf_dataloader

    def __iter__(self):
        for d in self.hf_dataloader:
            yield d['input_ids'], d['labels']
            # yield d['input_ids'][:-1], d['input_ids'][1:]
