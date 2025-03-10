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


import time
import matplotlib.pyplot as plt


def benchmark_layer(layer, x, num_runs=100):
    """Measure average forward pass time for a layer."""
    # Warmup
    for _ in range(10):
        _ = layer(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    # Timing
    start = time.time()
    for _ in range(num_runs):
        _ = layer(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()

    return (end - start) / num_runs * 1000  # Time in milliseconds


if __name__ == "__main__":
    # Device setup
    device = "mps"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # Parameters
    n = 512
    batch_sizes = [1, 16, 64, 256]
    num_runs = 100

    # Initialize layers
    cayley_linear = CayleyLinear(n, bias=True).to(device)
    standard_linear = nn.Linear(n, n, bias=True).to(device)

    # Sync weights for correctness check
    with torch.no_grad():
        W = cayley_linear.weight_matrix()
        standard_linear.weight.copy_(W.t())
        standard_linear.bias.copy_(cayley_linear.bias)

    # Validate correctness
    x_test = torch.randn(2, n, device=device)
    with torch.no_grad():
        y_standard = standard_linear(x_test)
        y_cayley = cayley_linear(x_test)
    diff_norm = torch.norm(y_standard - y_cayley).item()
    print(f"Output difference norm: {diff_norm:.2e} (should be near zero)")

    # Benchmarking
    cayley_times = []
    standard_times = []
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, n, device=device)

        cayley_time = benchmark_layer(cayley_linear, x, num_runs)
        standard_time = benchmark_layer(standard_linear, x, num_runs)

        cayley_times.append(cayley_time)
        standard_times.append(standard_time)

        print(f"Batch size {batch_size}:")
        print(f"  CayleyLinear: {cayley_time:.3f} ms")
        print(f"  Standard Linear: {standard_time:.3f} ms")

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(batch_sizes, standard_times, label="Standard Linear", marker="o", color="blue")
    plt.plot(batch_sizes, cayley_times, label="CayleyLinear (SO(n))", marker="o", color="orange")
    plt.xlabel("Batch Size")
    plt.ylabel("Average Forward Pass Time (ms)")
    plt.title(f"Speed Comparison: Standard Linear vs CayleyLinear (dim={n})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("speed_comparison.png")
    plt.show()
