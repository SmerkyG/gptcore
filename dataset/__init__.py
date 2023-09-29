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

@dataclass
class DM(lightning.LightningDataModule):
    dataset_path:str
    tokenizer_factory:typing.Callable
    sequence_length:int
    batch_size:int=1
    num_workers:int=0
    def get_dataloader(self, ds: Dataset, shuffle: bool = False):
        shuffle &= not isinstance(ds, IterableDataset)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers, pin_memory=True, collate_fn=collate_target_tokens_offset_by_one)
    
    # split can be train, validation, or test
    def get_dataset(self, split):
        tokenizer = self.tokenizer_factory()
        ds = datasets.load_dataset(path=self.dataset_path, streaming=True, split=split)
        ds = PipedDatasetWrapper(ds)
        ds = ds.map_batches(lambda input_batch: dataset.tokenizer.tokenize_join_and_slice(input_batch=input_batch, tokenizer=tokenizer, block_size=self.sequence_length), batch_size=4)
        if split == 'validation':
            ds = ds.take(max_count=1024)
        ds = ds.shuffle() # FIXME - does shuffle happen if we pass shuffle=False to the dataloader?
        return self.get_dataloader(ds, split == 'train')

    def train_dataloader(self): return self.get_dataset('train')
    def val_dataloader(self): return self.get_dataset('validation')

# def DM(dataset_path:str, tokenizer_factory:typing.Callable, sequence_length:int, batch_size:int=1, num_workers:int=0):
#     tokenizer = tokenizer_factory()
#     return lightning.LightningDataModule.from_datasets(
#         train_dataset=PipedDatasetWrapper(dataset=datasets.load_dataset(path=dataset_path, streaming=True, split='train'))
#             .map_batches(fn = lambda input_batch: dataset.tokenizer.tokenize_join_and_slice(input_batch=input_batch, tokenizer=tokenizer, block_size=sequence_length), batch_size=4)
#             .shuffle(),
#         val_dataset=PipedDatasetWrapper(dataset=datasets.load_dataset(path=dataset_path, streaming=True, split='validation'))
#             .map_batches(fn = lambda input_batch: dataset.tokenizer.tokenize_join_and_slice(input_batch=input_batch, tokenizer=tokenizer, block_size=sequence_length), batch_size=4)
#             .take(max_count=1024),
#         test_dataset=PipedDatasetWrapper(dataset=datasets.load_dataset(path=dataset_path, streaming=True, split='test'))
#             .map_batches(fn = lambda input_batch: dataset.tokenizer.tokenize_join_and_slice(input_batch=input_batch, tokenizer=tokenizer, block_size=sequence_length), batch_size=4),
#         batch_size=batch_size,
#         num_workers=num_workers,
#     )
