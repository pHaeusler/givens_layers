import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

from givens_linear import GivensLinear
from givens_linear_parallel import GivensLinearParallel
from cayley_linear import CayleyLinear
from clifford_linear import CliffordLinear
from exponential_linear import ExponentialLinear
from householder_linear import HouseholderLinear


def benchmark_layer(layer, x, num_runs=100):
    """Measure average forward pass time in milliseconds."""
    for _ in range(3):  # Warm-up runs
        _ = layer(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        _ = layer(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.time()
    return (end - start) / num_runs * 1000  # Convert to ms


if __name__ == "__main__":
    # Device and dtype setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    print(f"Running on {device} with dtype {dtype}")

    n = 1024
    batch_sizes = [1, 16, 64, 256, 1024]
    num_runs = 10

    layers = {
        "Standard Linear": nn.Linear(n, n, bias=True).to(device, dtype=dtype),
        "GivensLinear": GivensLinear(n, bias=True).to(device, dtype=dtype),
        "GivensLinearParallel": GivensLinearParallel(n, bias=True).to(device, dtype=dtype),
        "CayleyLinear (inv)": CayleyLinear(n, bias=True, method="inv").to(device, dtype=dtype),
        "CayleyLinear (cholesky)": CayleyLinear(n, bias=True, method="cholesky").to(
            device, dtype=dtype
        ),
        "CayleyLinear (neumann)": CayleyLinear(n, bias=True, method="neumann").to(
            device, dtype=dtype
        ),
        "CayleyLinear (qr)": CayleyLinear(n, bias=True, method="qr").to(device, dtype=dtype),
        "CayleyLinear (approx)": CayleyLinear(n, bias=True, method="approx").to(
            device, dtype=dtype
        ),
        "CliffordLinear": CliffordLinear(n, bias=True).to(device, dtype=dtype),
        "ExponentialLinear": ExponentialLinear(n, bias=True).to(device, dtype=dtype),
        "HouseholderLinear": HouseholderLinear(n, bias=True).to(device, dtype=dtype),
    }

    # Benchmarking
    results = {name: [] for name in layers}
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, n, device=device, dtype=dtype)
        print(f"\nBatch size {batch_size}:")
        for name, layer in layers.items():
            time_ms = benchmark_layer(layer, x, num_runs)
            results[name].append(time_ms)
            print(f"  {name}: {time_ms:.3f} ms")

    # Plotting results
    plt.figure(figsize=(12, 7))
    for name, times in results.items():
        plt.plot(batch_sizes, times, label=name, marker="o")
    plt.xlabel("Batch Size")
    plt.ylabel("Average Forward Pass Time (ms)")
    plt.title(f"Speed Comparison: Linear Layers (dim={n})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("speed_comparison.png")
    print("\nPlot saved as 'speed_comparison.png'")
