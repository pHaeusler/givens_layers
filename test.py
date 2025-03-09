import torch
import math
from givens_linear import GivensLinear, weight_matrix
from givens_linear_parallel import GivensLinearParallel

n = 32

gl = GivensLinear(n, bias=True)
gl_parallel = GivensLinearParallel(n, bias=True)

with torch.no_grad():
    W_gl = weight_matrix(gl.angles, gl.pairs, n)
    gl_parallel.angles.copy_(gl.angles)
    gl_parallel.bias.copy_(gl.bias)

print("W_gl:", W_gl.shape)

linear = torch.nn.Linear(n, n, bias=True)
with torch.no_grad():
    linear.weight.copy_(W_gl.t())
    linear.bias.copy_(gl.bias)

batch_size = 2
x = torch.randn(batch_size, n)
y_linear = linear(x)
y_gl = gl(x)
y_gl_parallel = gl_parallel(x)
print("Output difference norm:", torch.norm(y_linear - y_gl).item())
print("Output difference norm parallel:", torch.norm(y_linear - y_gl_parallel).item())
