import torch
import torch.nn as nn


def cayley_transform(A, method="inv"):
    I = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    I_minus_A = I - A
    I_plus_A = I + A

    if method == "inv":
        W = torch.linalg.solve(I_minus_A, I_plus_A)
    elif method == "cholesky":
        L = torch.linalg.cholesky(I_minus_A)
        W = torch.cholesky_solve(I_plus_A, L)
    elif method == "neumann":
        A2 = A @ A
        inv_approx = I + A + A2
        W = inv_approx @ I_plus_A
    elif method == "qr":
        Q, R = torch.linalg.qr(I_minus_A)
        W = torch.linalg.solve_triangular(R, Q.T @ I_plus_A, upper=True)
    elif method == "approx":
        W = I + 2 * A
    else:
        raise ValueError(f"Unknown method: {method}")

    return W


class CayleyLinear(nn.Module):
    def __init__(self, dim, bias=True, method="inv"):
        super(CayleyLinear, self).__init__()
        self.method = method
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
        output = x @ self.weight_matrix()
        return output + self.bias if self.bias is not None else output

    def weight_matrix(self):
        return cayley_transform(self._construct_skew_symmetric(), self.method)
