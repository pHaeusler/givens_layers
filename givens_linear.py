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

    def forward(self, x):
        assert x.shape[-1] == self.dim
        output = x.clone()
        cos_vals, sin_vals = torch.cos(self.angles), torch.sin(self.angles)
        for idx, (i, j) in enumerate(self.pairs):
            cos_theta, sin_theta = cos_vals[idx], sin_vals[idx]
            xi, xj = output[..., i].clone(), output[..., j].clone()
            output[..., i] = cos_theta * xi - sin_theta * xj
            output[..., j] = sin_theta * xi + cos_theta * xj
        return output + self.bias if self.bias is not None else output

    def weight_matrix(self):
        W = torch.eye(self.dim, device=self.angles.device)
        cos_vals, sin_vals = torch.cos(self.angles), torch.sin(self.angles)
        for idx, (i, j) in enumerate(self.pairs):
            cos_theta, sin_theta = cos_vals[idx], sin_vals[idx]
            Wi, Wj = W[:, i].clone(), W[:, j].clone()
            W[:, i] = cos_theta * Wi - sin_theta * Wj
            W[:, j] = sin_theta * Wi + cos_theta * Wj
        return W


if __name__ == "__main__":
    n = 512
    batch_size = 2
    x = torch.randn(batch_size, n)
    linear = torch.nn.Linear(n, n, bias=True)

    gl = GivensLinear(n, bias=True)
    with torch.no_grad():
        W = gl.weight_matrix()

    with torch.no_grad():
        linear.weight.copy_(W.t())
        linear.bias.copy_(gl.bias)

    with torch.no_grad():
        y_linear = linear(x)
        y_gl = gl(x)
    diff_norm = torch.norm(y_linear - y_gl).item()

    print("Output difference norm:", diff_norm)
