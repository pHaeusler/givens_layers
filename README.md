<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>

# Orthogonal Weight Matrices for Transformers

Reading about the Muon optimizer got me thinking.

Muon uses Newton-Schulz iterations to orthogonalize weight updates during each optimization step. This softly encourages orthongonality in the weight matrices of transformers.

The results are faster convergence and improved stability.

This isn’t the first time researchers have tried introduce orthogonality. Some have used Singular Value Decomposition (SVD) to enforce it, while others have added explicit constraints on the weight matrices.

$A^T A - I = 0$

For example, this can be applied to a linear layer with an additional loss term:

```python
dim = 512
linear = nn.Linear(dim, dim)
orthogonal_loss = linear.data.T @ linear.data - torch.eye(dim)
```

Can we avoid the optimization approach, and instead use a parameterization of $SO(n)$ that is efficient and easy to compute?

I.e. rather than storing $n^2$ parameters, can we store the minimum number of parameters for an orthogonal matrix?

and, rather than multiplying the input by the full matrix, can we use a more efficient algorithm for this particular parameterization?

## Why do we want orthogonal weight matrices?

Unconstrained weight matrices can rotate, scale, and skew inputs. Obviously this works, but is it an overparameterization?

Do we need to scale and skew the token features in a transformer? Is it enought to just rotate them?

The Muon optimizer suggests that rotation is enough.

Let's consider some definitions: $O(n)$ and $SO(n)$.

$O(n)$ is the set of all matrices that satisfy $A^T A = I$. This includes all rotations and reflections. The determinant is ±1 preserving the norm of the input. When the determinant is -1, the matrix is a reflection.

$SO(n)$ is the set of all matrices that satisfy $A^T A = I$ and $\det(A) = 1$. This restricts to pure rotations.

For both, the condition number is 1, which provides numerical stability - very helpful for training deep networks.

Let's consider the key and value matrices in a transformer. Tokens in a sequence are independently mapped to $k = x W_K$ and $v = x W_V$ before attention.

If these weight matrices are orthogonal, they rotate each token’s features without scaling or skewing, ensuring:

- **Independent Dimensions**: features are rotated independently, maintaining distinct representations.
- **Stability**: no skewing of features, so `nn.LayerNorm` is potentially redundant
- **Parameter Efficiency**: $SO(n)$ minimally requires $\frac{n(n-1)}{2}$ parameters, compared to $n^2$ for full matrices.
- **FLOPs Efficiency**: If a suitable parameterization of $SO(n)$ is found, the number of FLOPs required to compute the rotation can potentially be reduced.

So is there an efficient way to parameterize $SO(n)$?

## Parameterization approaches for $SO(n)$

Let's consider some approaches
- Givens Rotations
- Clifford Algebras (rotors)
- Exponential Map (Lie algebra)
- Cayley Transform (skew-symmetric)
- Householder Reflections

### Givens Rotations

Givens rotations parameterize $SO(n)$ with $\frac{n(n-1)}{2}$ angles, each tied to a pair of dimensions.

The number of angles is due to the number of unique pairs of dimensions in an $n$-dimensional space.

A Givens rotation is a 2d rotation matrix, positioned at specific $ij$ indices within a larger ($n \times n$) identity matrix.

$$
G_{ij} = \begin{bmatrix}
    1 & 0 & 0 & \cdots & 0 \\
    0 & \cos \theta & -\sin \theta & \cdots & 0 \\
    0 & \sin \theta & \cos \theta & \cdots & 0 \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    0 & 0 & 0 & \cdots & 1
\end{bmatrix}
$$

Any higher dimensional $SO(n)$ matrix can be built by multiplying 
$\frac{n(n-1)}{2}$ rotation matrices, each rotating the input in a different plane.

Given an $n$-dimensional weight matrix $W$, we can define all planes as:

```python
pairs = [(i, j) for i in range(dim) for j in range(i + 1, dim)]
```

Rather than materializing each sparse rotation matrix and multiplying them together, the combined matrix can be computed in place with:

```python
def weight_matrix(angles, pairs, dim):
    W = torch.eye(dim, device=angles.device)
    cos_vals, sin_vals = torch.cos(angles), torch.sin(angles)
    for idx, (i, j) in enumerate(pairs):
        cos_theta, sin_theta = cos_vals[idx], sin_vals[idx]
        Wi, Wj = W[:, i].clone(), W[:, j].clone()
        W[:, i] = cos_theta * Wi - sin_theta * Wj
        W[:, j] = sin_theta * Wi + cos_theta * Wj
    return W
```

With this, we can build a linear layer with that enforces orthogonality.

Here is a drop in replacement for standard `nn.Linear` layer for vector (not matrix) inputs.

```python
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
```

**Comparing Parameters**

This linear layer takes $\frac{n(n-1)}{2}$ parameters, compared to $n^2$ for a full matrix.

$$
\frac{n(n-1)}{2} / n^2 = \frac{n-1}{2n} \approx 49.9\%
$$

This is a reduction of 50% in parameters.

**Comparing FLOPs**

The classic linear layer (for vector inputs):

$$
y=xA+b
$$

This requires a vector-matrix multiplication ($xA$), which is O(n²) FLOPs.

This is the same as the Givens linear layer.

However, the Givens layer *must* be computed sequentially through the pairs of dimensions, whereas vector-matrix multiplication is highly parallel.

As a result, the Givens layer is significantly slower than the classic linear layer on a GPU.

**Potential Optimizations**

GivensLinear can be optimized by considering that disjoint rotations can be applied in parallel. So, careful parallel scheduling of rotations reduces the step count to O(n), with $\lfloor \frac{n}{2} \rfloor$ rotations per step. However, total FLOPs remain approximately O(n²).

Therefore, Givens composition is parameter-efficient but computationally intensive.

**Crazy Optimizations**

Since the dimensions are fixed, the sequence of rotations is also fixed. So it's possible to symbolically compute the full rotation matrix, and apply it in one go.

This is a reduction in FLOPs to O(n), at the cost of an insane pre-computed expression.

For example, using sympy

```python
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
```

This results in the following matrix:
$$
\begin{bmatrix}
    \cos(\theta_1) \cos(\theta_2) & -\sin(\theta_1) \cos(\theta_3) - \sin(\theta_2) \sin(\theta_3) \cos(\theta_1) & \sin(\theta_1) \sin(\theta_3) - \sin(\theta_2) \cos(\theta_1) \cos(\theta_3) \\
    \sin(\theta_1) \cos(\theta_2) & -\sin(\theta_1) \sin(\theta_2) \sin(\theta_3) + \cos(\theta_1) \cos(\theta_3) & -\sin(\theta_1) \sin(\theta_2) \cos(\theta_3) - \sin(\theta_3) \cos(\theta_1) \\
    \sin(\theta_2) & \sin(\theta_3) \cos(\theta_2) & \cos(\theta_2) \cos(\theta_3)
\end{bmatrix}
$$

This is a reduction in FLOPs to O(n), at the cost of an insane pre-computed expression.

This problem is that solving for higher dimensions (n=64, or 512) is intractable, although enticing...

## Clifford Algebras

Clifford algebras, $Cl(n)$, offer a way to manage rotations in $n$-dimensional space.

It is a system that fuses numbers and directions into a framework.

Consider a space with $n$ directions, like $x, y, z$ in 3D. These are defined by basis vectors: $e_1$ (a unit step along $x$), $e_2$ (along $y$), up to $e_n$. 

These vectors follow multiplication rules:
- $e_i e_i = 1$ (square to 1)
- $e_i e_j = -e_j e_i$ (anti-commutative)
- $e_i \cdot e_j = 0$ (perpendicular)

This setup was crafted by William Clifford in the 1870s, combining quaternions and Grassmann algebras. His aim was a system capturing space’s geometry, defining how lines, planes, and volumes interact.

From these rules, $Cl(n)$ generates:
- scalars
- vectors (arrows like $3 e_1 + 2 e_2$),
- bivectors (rotating planes like $e_1 e_2$)

Now, how do we rotate things in this system?

Introducing a rotor, $R$.

A rotor is a special object from $Cl(n)$ designed to spin vectors without stretching them.

A single-plane rotor is:
$$
R_{ij} = \cos(\theta/2) + \sin(\theta/2) e_i e_j
$$

This is a multivector, blending numbers and plane-spinning terms, and it rotates a vector $x$ into $x'$ via:

$$
R x \tilde{R} = x'
$$

This keeps $x$’s length unchanged, aligning with $SO(n)$’s rule for pure rotations.

A general rotor combines these across $\frac{n(n-1)}{2}$ planes:

$$
R = \prod_{k=1}^{\frac{n(n-1)}{2}} [\cos(\theta_k/2) + \sin(\theta_k/2) B_k]
$$

Where $B_k$ is a bivector defined as $e_i e_j$ for some $i, j$ pair.

Comparison to Givens Rotations

Parameters: $\frac{n(n-1)}{2}$ angles

Implementation Approaches

- Givens:
  - Dense: Precompute $G$
  - Sparse: Apply each rotation to $x$
- Clifford:
  - Explicit: Store angles and bivector indices; compute $Rx\tilde{R}$
  - Simplified: Use angles as Givens rotations ($O(n^2)$ memory).

Gradient Computation
- Givens:
  - Dense: Backprop through precomputed $G$ (standard autograd, $O(n^2)$ per sample).
  - Sparse: Chain rule through each rotation; update angles via $\frac{\partial L}{\partial x} \cdot \frac{\partial x}{\partial \theta}$ (e.g., $\frac{\partial x_i}{\partial \theta} = -\sin\theta x_i - \cos\theta x_j$), $O(n^2)$
- Clifford:
  - Explicit: Differentiate $Rx\tilde{R}$ in $Cl(n)$ (custom gradients, $O(n^2)$ with library like clifford).
  - Simplified: Treat as Givens; backprop through sequential rotations ($O(n^2)$), PyTorch-friendly.

Clifford rotors generalize elegantly but align with Givens in Euclidean tasks.

### Exponential Map

Let's talk about [Lie algebra](https://en.wikipedia.org/wiki/Lie_algebra) and [Lie groups](https://en.wikipedia.org/wiki/Lie_group).

Both $O(n)$ and $SO(n)$ are Lie groups. The first is the Orthogonal group, and the second is the Special Orthogonal group. They define manifolds of matricies based on constraints/conditions.

We're aiming to parameterise a linear layer weight matrix in the Special Orthogonal group $W \in SO(n)$.

The Lie algebra of a Lie group is the tangent space. Its the local linearization of the group. Think of it like a local flat surface where a point on a complex curved manifold can be described as a linear transformation.

Suppose you have a weight matrix $W \in SO(n)$.

To ensure this matrix stays in the $SO(n)$ group during optimization, the lie algebra is a way to project the gradient matrix $\nabla_W L$ onto the tangent space of $SO(n)$.

Fortunately this is well known for $SO(n)$ with Riemannian optimization on matrix manifolds.

To do this:
- Compute the skew-symmetric component: $G = \frac{1}{2} (W^T \nabla_W L - (\nabla_W L)^T W)$
- Apply the exponential map to get the tangent space: $W_{update} = \exp(\beta G)$

This $G$ is the “direction” of the update in the tangent space at $W$.

**Practical Considerations**
- Storage:
  - $W$ is a full $n \times n$ matrix. We must store and update the full matrix, relying on the projection to enforce the manifold.
- FLOPs:
  - Gradient projection: O(n³)
  - Exponential: O(n³) - this generally requires an eigenvalue decomposition

So unfortunately, Lie algebras is elegant, but computationally heavy.

Maybe this is an interesting alternative to the Muon Newton-Schulz iterations?

## Cayley Transform

The Cayley Transform offers a compelling approach to parameterize $SO(n)$. It maps a skew-symmetric matrix $A$ (where $A^T = -A$) to an orthogonal matrix via:

$W = (I - A)^{-1}(I + A)$

Here, $I$ is the identity matrix, and $A$ is a skew-symmetric $n \times n$ matrix. This transformation ensures $W \in SO(n)$ (with determinant 1), provided $I - A$ is invertible, which holds as long as $-1$ is not an eigenvalue of $A$.

The Cayley Transform is parameter-efficient. A skew-symmetric matrix $A$ has zeros on the diagonal and $\frac{n(n-1)}{2}$ independent entries above the diagonal (the lower triangle is determined by $A_{ji} = -A_{ij}$). This matches the minimal number of parameters required for $SO(n)$.

It provides a direct mapping from a compact set of parameters to the $SO(n)$ manifold, avoiding the iterative composition of Givens rotations or the heavy eigenvalue computations of the exponential map.

Think of $A$ as encoding the “amount” of rotation in each plane. The transform turns this into a rotation matrix $W$ that preserves norms. For small $A$, the Cayley Transform approximates the exponential map:

$W \approx I + 2A \quad (\text{for small } A)$

During training, optimize the parameters of $A$ directly. The gradient flows through the $W$ transform, and since $W$ is guaranteed to be in $SO(n)$, no additional constraints or projections are needed.

**Practical Considerations**

- Storage:
  - Store the $\frac{n(n-1)}{2}$ upper-triangular elements of a skew-symmetric matrix. This is the minimal number of parameters for $SO(n)$.
- FLOPs:
  - To construct $W$ from $A$ we need to compute $(I - A)^{-1}(I + A)$.
    - Matrix Inversion: $O(n^3)$
    - Matrix Multiplications: $O(n^3)$ (parallelizable on the GPU)

This is a drawback is that the inversion cost of $O(n^3)$ is no better than the exponential map or SVD-based methods.

However, there are optimizations to explore:
- Approximate the Cayley Transform using a Taylor series expansion
- cuSOLVER has a solver to speed up the inversion on GPUs
- Sparse $A$: inversion could drop to $O(n)$ using specialized solvers.
- Precomputation: If $A$ updates slowly, reuse $W$ over multiple steps.

### Householder Reflections

An $SO(n)$ matrix can be factored into up to $n$ Householder reflections, each defined by an $n$-dimensional vector, totaling $n^2$ parameters.

Applying $m$ reflections costs $O(mn)$ flops, but spanning $SO(n)$ demands $m \approx n$, yielding $O(n^2)$. Reducing $m$ restricts coverage.




### Alternative Methods

**Cayley Transform**: Maps skew-symmetric (A) to $(I - A)^{-1}(I + A)$, using $\frac{n(n-1)}{2}$ parameters, but requires an O(n³) inversion.

**Clifford Algebras**: Rotors generalize quaternions, yet retain $\frac{n(n-1)}{2}$ parameters and O(n²) application flops.

**SO(4) Dual Quaternions**: For $n = 4$, SO(4) leverages two quaternions (6 parameters), with O(n) flops, but this does not extend efficiently beyond $n = 4$.

For large ( n ), no method rivals quaternions’ SO(3) efficiency; full SO(n) resists lightweight parameterization.

