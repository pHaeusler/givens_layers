import torch
import torch.nn as nn


class ExponentialLinear(nn.Module):
    def __init__(self, dim, bias=True):
        super(ExponentialLinear, self).__init__()
        self.dim = dim
        self.num_params = (dim * (dim - 1)) // 2
        self.upper_indices = torch.triu_indices(dim, dim, offset=1)
        self.angles = nn.Parameter(torch.randn(self.num_params) * 0.01)
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def _construct_skew_symmetric(self):
        A = torch.zeros(self.dim, self.dim, device=self.angles.device, dtype=self.angles.dtype)
        A[self.upper_indices[0], self.upper_indices[1]] = self.angles
        A[self.upper_indices[1], self.upper_indices[0]] = -self.angles
        return A

    def forward(self, x):
        output = x @ self.weight_matrix()
        return output + self.bias if self.bias is not None else output

    def weight_matrix(self):
        A = self._construct_skew_symmetric()
        return torch.linalg.matrix_exp(A)
