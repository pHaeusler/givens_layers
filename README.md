# Orthogonal Weight Matrices for Transformers

Reading about the Muon optimizer got me thinking.

Muon uses Newton-Schulz iterations to orthogonalize weight updates during each optimization step. This softly encourages orthongonality in the weight matrices of transformers.

The results are faster convergence and improved stability.

This isn’t the first time researchers have tried introduce orthogonality. Some have used Singular Value Decomposition (SVD) to enforce it, while others have added explicit constraints on the weight matrices.

$A^T A - I = 0$

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

## Parameterizations of SO(n)

Let's consider some approaches.

### Givens Rotations

Givens rotations parameterize SO(n) with $\frac{n(n-1)}{2}$ angles, each tied to a pair of dimensions.

A Givens rotation is a 2d rotation matrix, positioned at specific ij indices within a larger ($n \times n$) identity matrix.

Specifically, any SO(n) matrix can be built by multiplying 
$\frac{n(n-1)}{2}$
 Givens rotations matrices. Each of these matrices, rotates the input in a different plane.

This requires O(n²) FLOPs.

This can be optimized by considering that disjoint rotations can be applied in parallel. So, careful parallel scheduling of rotations reduces the step count to O(n), with $\lfloor \frac{n}{2} \rfloor$ rotations per step. However, total FLOPs remain O(n²).

Therefore, Givens composition is parameter-efficient but computationally intensive.

There are some examples of these in the repo
- `GivensLinear`
- `GivensLinearParallel`

These are drop in replacement for standard `nn.Linear` layers, but enforce orthogonality.

### Exponential Map

Let's introduce [Lie algebra](https://en.wikipedia.org/wiki/Lie_algebra) and [Lie group](https://en.wikipedia.org/wiki/Lie_group) theory.

Both $O(n)$ and $SO(n)$ are Lie groups. The first is the Orthogonal group, and the second is the Special Orthogonal group. They define manifolds of matricies based on constraints/conditions.

The Lie algebra of a Lie group is the tangent space. Its the local linearization of the group. Think of it like a local flat surface where a point on a complex curved manifold can be described as a linear transformation.

Suppose you have a weight matrix $W \in SO(n)$.

To ensure this matrix stays in the SO group during optimization, the lie algebra is a way to project the gradient matrix $\nabla_W L$ onto the tangent space of $SO(n)$.

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

### Householder Reflections

An $SO(n)$ matrix can be factored into up to $n$ Householder reflections, each defined by an $n$-dimensional vector, totaling $n^2$ parameters.

Applying $m$ reflections costs $O(mn)$ flops, but spanning $SO(n)$ demands $m \approx n$, yielding $O(n^2)$. Reducing $m$ restricts coverage.


### Alternative Methods

**Cayley Transform**: Maps skew-symmetric (A) to $(I - A)^{-1}(I + A)$, using $\frac{n(n-1)}{2}$ parameters, but requires an O(n³) inversion.

**Clifford Algebras**: Rotors generalize quaternions, yet retain $\frac{n(n-1)}{2}$ parameters and O(n²) application flops.

**SO(4) Dual Quaternions**: For $n = 4$, SO(4) leverages two quaternions (6 parameters), with O(n) flops, but this does not extend efficiently beyond $n = 4$.

For large ( n ), no method rivals quaternions’ SO(3) efficiency; full SO(n) resists lightweight parameterization.

