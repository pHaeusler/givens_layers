import torch
import torch.nn as nn


class _HouseholderLinear(nn.Module):
    def __init__(self, dim, num_reflections=None, bias=True):
        super(HouseholderLinear, self).__init__()
        self.dim = dim
        self.m = num_reflections if num_reflections is not None else dim
        self.vectors = nn.Parameter(torch.randn(self.m, dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x):
        assert x.shape[-1] == self.dim
        output = x.clone()
        v_normed = self.vectors / torch.norm(self.vectors, p=2, dim=1, keepdim=True)
        for i in range(self.m):
            v = v_normed[i]
            proj = torch.einsum("...d,d->...", output, v)  # Dot product
            output = output - 2 * torch.einsum("...,d->...d", proj, v)
        return output + self.bias if self.bias is not None else output

    def weight_matrix(self):
        v_normed = self.vectors / torch.norm(self.vectors, p=2, dim=1, keepdim=True)
        W = torch.eye(self.dim, device=self.vectors.device)
        for i in range(self.m):
            v = v_normed[i]
            Wv = W @ v
            W = W - 2 * torch.outer(Wv, v)
        return W


class HouseholderLinear(nn.Module):
    def __init__(self, dim, num_reflections=None, bias=True):
        super(HouseholderLinear, self).__init__()
        self.dim = dim
        self.m = num_reflections if num_reflections is not None else dim
        self.vectors = nn.Parameter(torch.randn(self.m, dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def _construct_householder_product(self):
        v_normed = self.vectors / torch.norm(self.vectors, p=2, dim=1, keepdim=True)
        A = v_normed.clone()
        if self.m < self.dim:
            A = torch.cat([A, torch.zeros(self.dim - self.m, self.dim, device=A.device)], dim=0)
        else:
            A = A[: self.dim, :]
        tau = torch.full((min(self.m, self.dim),), 2.0, device=A.device, dtype=A.dtype)
        Q = torch.linalg.householder_product(A, tau)
        return Q[:, : self.dim]

    def forward(self, x):
        assert x.shape[-1] == self.dim
        Q = self._construct_householder_product()
        output = x @ Q
        return output + self.bias if self.bias is not None else output

    def weight_matrix(self):
        return self._construct_householder_product()


if __name__ == "__main__":
    dim = 32
    layer = HouseholderLinear(dim, dim)
    linear = nn.Linear(dim, dim, bias=True)
    with torch.no_grad():
        linear.weight.copy_(layer.weight_matrix().t())
        linear.bias.copy_(layer.bias)
    x = torch.randn(32, dim)
    print(torch.norm(layer(x) - linear(x)))
