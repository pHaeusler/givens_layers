import torch
import torch.nn as nn


def cayley_transform_approx(A, order=1):
    """
    Approximate Cayley Transform: W ≈ (I + A + ... + A^order) * (I + A) for small A, reducing complexity to O(n^2).
    """
    dim = A.shape[-1]
    I = torch.eye(dim, device=A.device, dtype=A.dtype)

    # Compute I + A
    I_plus_A = I + A  # O(n^2)

    # Approximate (I - A)^(-1) with Taylor series
    inv_approx = I.clone()
    A_power = A.clone()
    for _ in range(order):
        inv_approx = inv_approx + A_power  # O(n^2) addition
        A_power = torch.matmul(A_power, A)  # O(n^2) per multiplication

    # Compute W ≈ (I - A)^(-1) * (I + A)
    W = torch.matmul(inv_approx, I_plus_A)  # O(n^2)

    return W


class CayleyLinearApprox(nn.Module):
    def __init__(self, dim, bias=True, order=1):
        super(CayleyLinearApprox, self).__init__()
        self.dim = dim
        self.order = order
        self.num_params = (dim * (dim - 1)) // 2
        self.angles = nn.Parameter(torch.randn(self.num_params) * 0.01)
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.upper_indices = [(i, j) for i in range(dim) for j in range(i + 1, dim)]
        assert len(self.upper_indices) == self.num_params

    def _construct_skew_symmetric(self):
        A = torch.zeros(self.dim, self.dim, device=self.angles.device, dtype=self.angles.dtype)
        for idx, (i, j) in enumerate(self.upper_indices):
            A[i, j] = self.angles[idx]
            A[j, i] = -self.angles[idx]
        return A

    def forward(self, x):
        assert x.shape[-1] == self.dim
        A = self._construct_skew_symmetric()
        W = cayley_transform_approx(A, order=self.order)
        output = torch.matmul(x, W)
        if self.bias is not None:
            output = output + self.bias
        return output


# Test the approximate version
if __name__ == "__main__":
    dim = 64
    model = CayleyLinearApprox(dim, order=16)
    x = torch.randn(64, dim)
    output = model(x)

    A = model._construct_skew_symmetric()
    W = cayley_transform_approx(A, order=1)
    print(
        f"Orthogonality error (approx, expect non-zero): {torch.norm(W.T @ W - torch.eye(dim)):.6f}"
    )
    print(f"Output shape: {output.shape}")
