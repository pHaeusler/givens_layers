import torch
import torch.nn as nn


def cayley_transform_exact(A):
    I = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    W = torch.linalg.solve(I - A, I + A)
    return W


class CayleyLinear(nn.Module):
    def __init__(self, dim, bias=True):
        super(CayleyLinear, self).__init__()
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
        assert x.shape[-1] == self.dim
        output = x @ cayley_transform_exact(self._construct_skew_symmetric())
        return output + self.bias if self.bias is not None else output

    def weight_matrix(self):
        return cayley_transform_exact(self._construct_skew_symmetric())


if __name__ == "__main__":
    n = 512
    batch_size = 2
    x = torch.randn(batch_size, n)
    linear = torch.nn.Linear(n, n, bias=True)

    cl = CayleyLinear(n, bias=True)
    with torch.no_grad():
        W = cl.weight_matrix()

    with torch.no_grad():
        linear.weight.copy_(W.t())
        linear.bias.copy_(cl.bias)

    with torch.no_grad():
        y_linear = linear(x)
        y_cl = cl(x)
    diff_norm = torch.norm(y_linear - y_cl).item()

    print("Output difference norm:", diff_norm)
