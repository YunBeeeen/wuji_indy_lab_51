from collections.abc import Sequence
import numpy as np
import torch
    
class TorchRunningStats:
    def __init__(self, dim=1, device="cpu"):
        self.n = torch.zeros(dim, dtype=torch.float, device=device)
        self.mu = torch.zeros(dim, dtype=torch.float, device=device)
        self.M2 = torch.zeros(dim, dtype=torch.float, device=device)
        self.dim = dim
        self.device = device

    def update(self, x):
        self.n += 1
        delta = x - self.mu
        self.mu += delta / self.n
        delta2 = x - self.mu
        self.M2 += delta * delta2

    def reset(self, ids: Sequence[int] | None = None):
        self.n[ids] = 0.0
        self.mu[ids] = 0.0
        self.M2[ids] = 0.0

    def count(self):
        return self.n

    def mean(self):
        return torch.where(self.n != 0, self.mu, 0.0)

    def variance(self):
        return torch.where(self.n > 1, self.M2 / (self.n - 1), 0.0)

    def standard_deviation(self):
        return torch.sqrt(self.variance())

###################################################
if __name__ == "__main__":
    time = 400
    dim = 1024
    running_stats = TorchRunningStats(dim=dim)
    data_stream = torch.randn(size=(time, dim))

    for t in range(time):
        running_stats.update(data_stream[t])

    true_mean = torch.mean(data_stream, dim=0)
    true_std = torch.std(data_stream, dim=0)
    computed_mean = running_stats.mean()
    computed_std = running_stats.standard_deviation()
    print(f"Count: {torch.norm(running_stats.count() - time) < 1e-3}")
    print(f"Mean: {torch.norm(running_stats.mean() - true_mean) < 1e-3}")
    print(f"Std: {torch.norm(running_stats.standard_deviation() - true_std) < 1e-3}")
