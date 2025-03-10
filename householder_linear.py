import torch
import torch.nn as nn


class HouseholderLinear(nn.Module):
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


if __name__ == "__main__":
    n = 512
    batch_size = 2
    x = torch.randn(batch_size, n)
    linear = torch.nn.Linear(n, n, bias=True)

    hl = HouseholderLinear(n, bias=True)
    with torch.no_grad():
        W = hl.weight_matrix()

    with torch.no_grad():
        linear.weight.copy_(W.t())
        linear.bias.copy_(hl.bias)

    with torch.no_grad():
        y_linear = linear(x)
        y_hl = hl(x)
    diff_norm = torch.norm(y_linear - y_hl).item()

    print("Output difference norm:", diff_norm)
