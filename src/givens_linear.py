import torch
import torch.nn as nn
import math


class GivensLinear(nn.Module):
    def __init__(self, dim, bias=True):
        super(GivensLinear, self).__init__()
        self.dim = dim
        self.pairs = [(i, j) for i in range(dim) for j in range(i + 1, dim)]
        self.angles = nn.Parameter(torch.randn(len(self.pairs)) * 0.01)
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def _compute(self, output):
        cos_vals, sin_vals = torch.cos(self.angles), torch.sin(self.angles)
        for idx, (i, j) in enumerate(self.pairs):
            output[:, [i, j]] = output[:, [i, j]] @ torch.tensor(
                [[cos_vals[idx], -sin_vals[idx]], [sin_vals[idx], cos_vals[idx]]],
                device=output.device,
                dtype=output.dtype,
            )
        return output

    def forward(self, x):
        assert x.shape[-1] == self.dim
        output = self._compute(x)
        return output + self.bias if self.bias is not None else output

    def weight_matrix(self):
        W = torch.eye(self.dim, device=self.angles.device)
        return self._compute(W)
