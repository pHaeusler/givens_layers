import torch
import torch.nn as nn


class ExponentialLinear(nn.Module):
    def __init__(self, dim, bias=True):
        super(ExponentialLinear, self).__init__()
        self.dim = dim
        self.num_params = (dim * (dim - 1)) // 2
        # Indices for upper triangular elements
        self.upper_indices = torch.triu_indices(dim, dim, offset=1)
        # Parameters for skew-symmetric matrix A
        self.angles = nn.Parameter(torch.randn(self.num_params) * 0.01)
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def _construct_skew_symmetric(self):
        # Build skew-symmetric matrix A from parameters
        A = torch.zeros(self.dim, self.dim, device=self.angles.device, dtype=self.angles.dtype)
        A[self.upper_indices[0], self.upper_indices[1]] = self.angles
        A[self.upper_indices[1], self.upper_indices[0]] = -self.angles
        return A

    def forward(self, x):
        A = self._construct_skew_symmetric()
        W = torch.linalg.matrix_exp(A)  # W is orthogonal by construction
        output = x @ W
        if self.bias is not None:
            output += self.bias
        return output


# Example usage
layer = ExponentialLinear(dim=3)
optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)

# Simulate a forward and backward pass
x = torch.randn(2, 3)
y = layer(x)
loss = y.sum()  # Dummy loss
loss.backward()
optimizer.step()

# Check orthogonality (W^T W should be identity)
W = torch.linalg.matrix_exp(layer._construct_skew_symmetric())
print("Orthogonality check:\n", W.T @ W)
