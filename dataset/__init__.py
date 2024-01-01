import numpy as np
import os
import torch
import torch.utils.data
import torchdata
import typing
import datasets

# class HuggingFaceDatasetWrapper(torch.utils.data.Dataset):
#     def __init__(self, hf_dataset : datasets.DatasetDict | datasets.Dataset | datasets.IterableDatasetDict | datasets.IterableDataset):
#         self.hf_dataset = hf_dataset

#     def __iter__(self):
#         for d in self.hf_dataset:
#             yield d

#     # def __len__(self):
#     #     return len(self.hf_dataset)

#     # def __getitem__(self, idx: int):
#     #     return self.hf_dataset[idx]

# class HuggingFacePipedDatasetWrapper(torchdata.datapipes.iter.IterDataPipe[typing.Any]):
#     def __init__(self, hf_dataset : datasets.DatasetDict | datasets.Dataset | datasets.IterableDatasetDict | datasets.IterableDataset):
#         self.hf_dataset = hf_dataset

#     def __iter__(self):
#         for d in self.hf_dataset:
#             yield d

import torch.utils.data.datapipes.datapipe

#T = typing.TypeVar("T")
T_co = typing.TypeVar('T_co', covariant=True)
class PipedDatasetWrapper(typing.Generic[T_co], torch.utils.data.datapipes.datapipe.IterDataPipe[T_co]):
#class PipedDatasetWrapper(torchdata.datapipes.iter.IterDataPipe[typing.Any]):
    def __init__(self, dataset : torch.utils.data.Dataset | datasets.DatasetDict | datasets.Dataset | datasets.IterableDatasetDict | datasets.IterableDataset):
        super().__init__()
        self.dataset = dataset

    def __iter__(self):
        for data in self.dataset:
            yield data

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

from typing import Iterator, Optional, TypeVar

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

#T_co = typing.TypeVar("T_co", covariant=True)

@functional_datapipe("take")
class TakeIterDataPipe(IterDataPipe[T_co]):
    """
    Yield at most the specified number of elements from the source DataPipe.

    Args:
        source_datapipe: source DataPipe that will be iterated through
        max_count: the maximum number of elements of ``source_datapipe`` yielded before the pipe ends

    Example:
        >>> from dataset import functional_datapipe
        >>> dp = functional_datapipe(range(3))
        >>> dp = dp.take(2)
        >>> list(dp)
        [0, 1]
    """

    def __init__(self, source_datapipe: IterDataPipe[T_co], max_count: int) -> None:
        self.source_datapipe: IterDataPipe[T_co] = source_datapipe
        self.max_count: int = max_count
        self.count: int = 0

    def __iter__(self) -> Iterator[T_co]:
        for element in self.source_datapipe:
            yield element
            self.count += 1
            if self.count >= self.max_count:
                return

    def __len__(self) -> int:
        return min(self.max_count, len(self.source_datapipe))


import lightning
import datasets
import dataset.tokenizer

from torch.utils.data import DataLoader, Dataset, IterableDataset
from dataclasses import dataclass, field

def collate_target_tokens_offset_by_one(batch): 
    values = torch.utils.data.default_collate(batch)
    return values[..., :-1], values[..., 1:]

def collate_target_tokens_offset_by_one_input_ids(batch): 
    tuple_batch = [(d['input_ids'][:-1], d['input_ids'][1:]) for d in batch]
    return torch.utils.data.default_collate(tuple_batch)

@dataclass
class DM(lightning.LightningDataModule):
    dataset_path:str
    tokenizer_factory:typing.Callable
    sequence_length:int
    batch_size:int=1
    num_workers:int=0
    seed:int|None=None

    def get_dataloader(self, ds: Dataset, shuffle: bool|None = None):
        return DataLoader(ds, 
                          #prefetch_factor=4, 
                          persistent_workers=True,
                          batch_size=self.batch_size, 
                          shuffle=shuffle, 
                          num_workers=min(ds.n_shards, self.num_workers), pin_memory=True, collate_fn=collate_target_tokens_offset_by_one_input_ids)

    def get_dataset(self, split):
        tokenizer = self.tokenizer_factory()
        ds = datasets.load_dataset(path=self.dataset_path, streaming=True, split=split)
        ds = ds.map(lambda x: dataset.tokenizer.tokenize_crop_join_and_slice_input_ids(x, tokenizer, self.sequence_length, 8 if split == 'train' else 1), batched=True, remove_columns=ds.column_names)
        #if split == 'train':
        ds = ds.shuffle(seed=self.seed)
        if split == 'validation':
            ds = ds.take(1024)
        return self.get_dataloader(ds, None)#None if split == 'train' else False)
    
    def train_dataloader(self): return self.get_dataset('train')
    def val_dataloader(self): return self.get_dataset('validation')

# FIXME - this version uses torchdata iterable pipes but doesn't have good results, in a way I can't manage to understand, maybe due to shuffling or something
# @dataclass
# class DM(lightning.LightningDataModule):
#     dataset_path:str
#     tokenizer_factory:typing.Callable
#     sequence_length:int
#     batch_size:int=1
#     num_workers:int=0
#     seed:int|None=None
#     def get_dataloader(self, ds: Dataset, shuffle: bool|None = None):
#         return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers, pin_memory=True, collate_fn=collate_target_tokens_offset_by_one, persistent_workers=True)
#     # split can be train, validation, or test
#     def get_dataset(self, split):
#         tokenizer = self.tokenizer_factory()
#         ds = datasets.load_dataset(path=self.dataset_path, streaming=True, split=split)
#         # if split == 'train':
#         #     ds = ds.shuffle() # FIXME - had to add this, somehow the pytorch shuffle isn't enough maybe because it doesn't know about the shards so can't shuffle shards?
#         ds = PipedDatasetWrapper(ds)
#         #if split == 'train':
#         #    ds = ds.shuffle(buffer_size=10000)
#         # join every 1000 texts together, tokenize them, cut that up into sequence_length chunks, then shuffle chunks within each buffer_size set of chunks
#         #torchdata.datapipes.iter.BatchMapper
#         ds = ds.map_batches(lambda input_batch: dataset.tokenizer.tokenize_join_and_slice(input_batch=input_batch, tokenizer=tokenizer, block_size=self.sequence_length), batch_size=1000)
#         if split == 'validation':
#             ds = ds.take(max_count=1024)
#         #torchdata.datapipes.iter.Shuffler
#         #torchdata.datapipes.iter.UnBatcher
#         ds = ds.shuffle(buffer_size=10000).set_seed(self.seed) # FIXME - does shuffle happen if we pass shuffle=False to the dataloader?
#         return self.get_dataloader(ds, None if split == 'train' else False)

#     def train_dataloader(self): return self.get_dataset('train')
#     def val_dataloader(self): return self.get_dataset('validation')
