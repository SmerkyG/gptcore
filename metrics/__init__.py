import torch
import torch.nn as nn
from torch import Tensor

from metrics.interface import IMetric, MetricArgs

class Accuracy(nn.Module, IMetric):
    def __init__(self):
        super().__init__()
        self.values = []

    def update(self, margs:MetricArgs):
        with torch.no_grad():
            self.values.append(margs.predictions.eq(margs.labels).sum() / (margs.labels.size(0)*margs.labels.size(1)))

    def compute(self):
        with torch.no_grad():
            value = sum(self.values) / len(self.values)
            self.values.clear()
            return value

class Loss(nn.Module, IMetric):
    def __init__(self):
        super().__init__()
        self.values = []

    def update(self, margs:MetricArgs):
        with torch.no_grad():
            self.values.append(margs.loss)

    def compute(self):
        with torch.no_grad():
            value = sum(self.values) / len(self.values)
            self.values.clear()
            return value
