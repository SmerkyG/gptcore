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
