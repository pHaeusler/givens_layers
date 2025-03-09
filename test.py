import time
import torch
import math
from givens_linear import GivensLinear, weight_matrix
from givens_linear_parallel import GivensLinearParallel


def time_forward(layer, x, num_runs=10):
    # Warm-up iterations
    for _ in range(10):
        _ = layer(x)
    start = time.time()
    for _ in range(num_runs):
        _ = layer(x)
    return (time.time() - start) / num_runs


def compare_givens_with_linear():
    n = 512
    batch_size = 2
    x = torch.randn(batch_size, n)
    linear = torch.nn.Linear(n, n, bias=True)

    # Compare GivensLinear approach
    gl = GivensLinear(n, bias=True)
    with torch.no_grad():
        W_gl = weight_matrix(gl.angles, gl.pairs, n)
    print("GivensLinear weight matrix shape:", W_gl.shape)

    with torch.no_grad():
        linear.weight.copy_(W_gl.t())
        linear.bias.copy_(gl.bias)

    linear_time = time_forward(linear, x)
    gl_time = time_forward(gl, x)

    with torch.no_grad():
        y_linear = linear(x)
        y_gl = gl(x)
    diff_norm = torch.norm(y_linear - y_gl).item()

    print("GivensLinear: Output difference norm:", diff_norm)
    print("GivensLinear: Linear forward average time: {:.6f} s".format(linear_time))
    print("GivensLinear: Givens forward average time: {:.6f} s".format(gl_time))

    # Compare GivensLinearParallel approach
    gl_parallel = GivensLinearParallel(n, bias=True)
    with torch.no_grad():
        W_gl_parallel = gl_parallel.weight_matrix()
    print("GivensLinearParallel weight matrix shape:", W_gl_parallel.shape)

    with torch.no_grad():
        linear.weight.copy_(W_gl_parallel.t())
        linear.bias.copy_(gl_parallel.bias)

    linear_time = time_forward(linear, x)
    gl_parallel_time = time_forward(gl_parallel, x)

    with torch.no_grad():
        y_linear = linear(x)
        y_gl_parallel = gl_parallel(x)
    diff_norm_parallel = torch.norm(y_linear - y_gl_parallel).item()

    print("GivensLinearParallel: Output difference norm:", diff_norm_parallel)
    print("GivensLinearParallel: Linear forward average time: {:.6f} s".format(linear_time))
    print("GivensLinearParallel: Givens forward average time: {:.6f} s".format(gl_parallel_time))


if __name__ == "__main__":
    compare_givens_with_linear()
