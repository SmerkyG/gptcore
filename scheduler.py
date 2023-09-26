import torch.optim.lr_scheduler

from dataclasses import dataclass

from util.config import Factory

from typing import Callable, Any, Optional, Tuple, List, Iterable

@dataclass
class LRSchedulerConfig:
    scheduler_factory : Callable[..., torch.optim.lr_scheduler.LRScheduler]
    interval:str = "epoch"
    frequency:int = 1
    monitor:str = "val/loss"
    strict:bool = True
    name:str|None = None

    def to_dict(self, optimizer):
        return dict(scheduler=self.scheduler_factory(optimizer), interval=self.interval, frequency=self.frequency, monitor=self.monitor, strict=self.strict, name=self.name)
    