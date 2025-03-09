# Orthogonal Weight Matricies for Transformers


- Givens rotations can perfectly represent any n-dimensional rotation SO(n)
- Butterfly FFT
- Blocks of Givens



## Efficient Composition with Minimal Operations
Composing n(n−1)2\frac{n(n-1)}{2}\frac{n(n-1)}{2}
Givens rotations naively (multiplying n×nn \times nn \times n
matrices) costs O(n3⋅n2)=O(n5)O(n^3 \cdot n^2) = O(n^5)O(n^3 \cdot n^2) = O(n^5)
operations, which is impractical. We need a faster way to apply the product W=G1G2⋯GmW = G_1 G_2 \cdots G_mW = G_1 G_2 \cdots G_m
to a vector or matrix.

Key Insight: Disjoint Pairs
Two Givens rotations can be applied simultaneously if they operate on disjoint pairs of coordinates—meaning their index pairs (i1,j1)(i_1, j_1)(i_1, j_1)
 and (i2,j2)(i_2, j_2)(i_2, j_2)
 have no indices in common (i.e., {i1,j1}∩{i2,j2}=∅\{i_1, j_1\} \cap \{i_2, j_2\} = \emptyset\{i_1, j_1\} \cap \{i_2, j_2\} = \emptyset
). For example:
Rotations on ( (0,1) ) and ( (2,3) ) are disjoint and can be applied in parallel.
Rotations on ( (0,1) ) and ( (0,2) ) are not disjoint (they both affect coordinate 0) and must be sequential.

In an ( n )-dimensional space:
We can apply up to ⌊n2⌋\lfloor \frac{n}{2} \rfloor\lfloor \frac{n}{2} \rfloor
 disjoint rotations in parallel in a single step, since each rotation involves two coordinates.

To cover all n(n−1)2\frac{n(n-1)}{2}\frac{n(n-1)}{2}
 possible pairs, we need multiple steps. The minimal number of such parallel steps corresponds to the edge coloring number of the complete graph KnK_nK_n
:
For even ( n ), we need n−1n-1n-1 steps.
For odd ( n ), we need ( n ) steps.


some papers are adding a orthogonal constriant (A^TA - I = 0) to the weight matrix to ensure that the weight matrix is orthogonal.