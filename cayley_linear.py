import torch
import torch.nn as nn


def cayley_transform_exact(A):
    I = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    I_minus_A, I_plus_A = I - A, I + A
    det_I_minus_A = torch.det(I_minus_A)
    adj_I_minus_A = torch.linalg.inv(I_minus_A).T * det_I_minus_A
    I_minus_A_inv = adj_I_minus_A / det_I_minus_A
    return torch.matmul(I_minus_A_inv, I_plus_A)


class CayleyLinear(nn.Module):
    def __init__(self, dim, bias=True):
        super(CayleyLinear, self).__init__()
        self.dim = dim
        self.num_params = (dim * (dim - 1)) // 2
        self.upper_indices = [(i, j) for i in range(dim) for j in range(i + 1, dim)]
        self.angles = nn.Parameter(torch.randn(self.num_params) * 0.01)
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        assert len(self.upper_indices) == self.num_params

    def _construct_skew_symmetric(self):
        A = torch.zeros(self.dim, self.dim, device=self.angles.device, dtype=self.angles.dtype)
        for idx, (i, j) in enumerate(self.upper_indices):
            A[i, j] = self.angles[idx]
            A[j, i] = -self.angles[idx]
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
