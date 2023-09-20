import torch
from torch import nn
import numpy as np

class TopKPTailFreeSampler(nn.Module):
    def __init__(self, temperature : float = 1.0, top_k : float | None = None, top_p : float | None = None, tail_free_sampling : float | None = None):
        super().__init__()
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.tail_free_sampling = tail_free_sampling
        
    def forward(self, x):
        x = torch.softmax(x, dim=-1)
        if self.top_k is not None and self.top_k > 0:
            x[x < torch.topk(x, k=self.top_k)[0][..., -1, None]] = 0
        if (self.top_p is not None and self.top_p > 0) or (self.tail_free_sampling is not None and self.tail_free_sampling > 0):
            sorted = x.sort(descending=True)[0]
            if self.tail_free_sampling is not None and self.tail_free_sampling > 0:
                self.top_p = 0.95
                d = sorted[1:] - sorted[:-1]
                d = d[1:] - d[:-1]
                d = d.abs()
                d = d / d.sum(dim=-1).item()
                cumulative_probs = d.cumsum(dim=-1).cpu().numpy()
            else:
                cumulative_probs = sorted.cumsum(dim=-1).cpu().numpy()
            cutoff = float(sorted[np.argmax(cumulative_probs > self.top_p)])
            x[x < cutoff] = 0
        if self.temperature != 1.0:
            x = x.pow(1.0 / self.temperature)
        x = torch.multinomial(x, num_samples=1)
        return x

