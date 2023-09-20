import torch
import torch.nn as nn
from torch import Tensor

from abc import abstractmethod

class IMetric():
    @abstractmethod
    def update(self, inputs, logits:Tensor, predictions:Tensor, labels:Tensor, loss:Tensor):
        raise NotImplementedError()

    @abstractmethod
    def compute(self):
        raise NotImplementedError()

class MetricArgs():
    def __init__(self, inputs, logits:Tensor, predictions:Tensor, labels:Tensor, loss:Tensor):
        with torch.no_grad():
            self.inputs = inputs.detach() if isinstance(inputs, Tensor) else inputs
            self.logits = logits.detach()
            self.predictions = predictions.detach()
            self.labels = labels.detach()
            self.loss = loss.detach()
