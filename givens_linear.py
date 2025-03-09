import torch
import torch.nn as nn
import math


def weight_matrix(angles, pairs, dim):
    W = torch.eye(dim, device=angles.device)
    cos_vals = torch.cos(angles)
    sin_vals = torch.sin(angles)
    for idx, (i, j) in enumerate(pairs):
        cos_theta = cos_vals[idx]
        sin_theta = sin_vals[idx]
        Wi = W[:, i].clone()
        Wj = W[:, j].clone()
        W[:, i] = cos_theta * Wi - sin_theta * Wj
        W[:, j] = sin_theta * Wi + cos_theta * Wj
    return W


class GivensLinear(nn.Module):
    def __init__(self, dim, bias=True):
        super(GivensLinear, self).__init__()
        self.dim = dim
        self.num_rotations = (dim * (dim - 1)) // 2
        self.angles = nn.Parameter(torch.randn(self.num_rotations) * 0.01)
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.pairs = []
        for i in range(dim):
            for j in range(i + 1, dim):
                self.pairs.append((i, j))
        assert len(self.pairs) == self.num_rotations

    def forward(self, x):
        assert x.shape[-1] == self.dim
        output = x.clone()
        cos_vals = torch.cos(self.angles)
        sin_vals = torch.sin(self.angles)
        for idx, (i, j) in enumerate(self.pairs):
            cos_theta = cos_vals[idx]
            sin_theta = sin_vals[idx]
            xi = output[..., i].clone()
            xj = output[..., j].clone()
            output[..., i] = cos_theta * xi - sin_theta * xj
            output[..., j] = sin_theta * xi + cos_theta * xj
        return output + self.bias if self.bias is not None else output
