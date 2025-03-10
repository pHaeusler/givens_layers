from sympy import SparseMatrix, cos, sin, symbols

n = 3
pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
thetas = symbols("theta1:%d" % (len(pairs) + 1))
W = SparseMatrix.eye(n)
for idx, (i, j) in enumerate(pairs):
    G = SparseMatrix.eye(n)
    theta = thetas[idx]
    G[i, i], G[j, j] = cos(theta), cos(theta)
    G[i, j], G[j, i] = -sin(theta), sin(theta)
    W = W * G

print(W)
