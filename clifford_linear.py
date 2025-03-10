import torch
import torch.nn as nn


class CliffordLinear(nn.Module):
    def __init__(self, dim, bias=True):
        super(CliffordLinear, self).__init__()
        self.dim = dim
        self.pairs = [(i, j) for i in range(dim) for j in range(i + 1, dim)]
        self.bivector_coeffs = nn.Parameter(torch.randn(len(self.pairs)) * 0.01)
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x):
        assert x.shape[-1] == self.dim
        output = x.clone()
        cos_vals = torch.cos(self.bivector_coeffs)
        sin_vals = torch.sin(self.bivector_coeffs)
        for idx, (i, j) in enumerate(self.pairs):
            cos_theta, sin_theta = cos_vals[idx], sin_vals[idx]
            xi, xj = output[..., i].clone(), output[..., j].clone()
            output[..., i] = cos_theta * xi - sin_theta * xj
            output[..., j] = sin_theta * xi + cos_theta * xj
        return output + self.bias if self.bias is not None else output
