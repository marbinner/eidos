# The idea

Coordinates are directions for a walk: $(2.9, 1.8)$ means 2.9 steps along $\hat\imath$, then 1.8 along $\hat\jmath$. Any two arrows not on one line can be the steps instead, and then the same arrow has other numbers and the same linear map another matrix. This scene converts between them, and shows that steps along the map's eigenvectors make its matrix as simple as it can be. You need the primers on span, linear maps (eigenvectors) and products of maps (inverses).

# The mathematics

## Coordinates in a basis

A basis of the plane is two arrows $b_1, b_2$, neither of length 0 and not on one line. Every arrow $v$ is then one walk along them, and only one:

$$ v = c_1 b_1 + c_2 b_2 $$

and $c = (c_1, c_2)$ are $v$'s coordinates in that basis. Let $B$ be the matrix with $b_1$ and $b_2$ as its columns. By the column rule of linear maps, the walk is $B$ applied to $c$:

$$ v = Bc, \qquad c = B^{-1} v $$

$B$ turns coordinates into the arrow; $B^{-1}$ turns it back. The inverse exists because the columns are not on one line, so $\det B \ne 0$.

Example: the scene opens with $b_1 = (1.2, 0.4)$, $b_2 = (0.5, 1)$ and $v = (2.9, 1.8)$. So $\det B = 1.2 \times 1 - 0.5 \times 0.4 = 1$, and

$$ B^{-1} = \begin{bmatrix} 1 & -0.5 \\ -0.4 & 1.2 \end{bmatrix}, \qquad B^{-1}v = (2.9 - 0.9,\ -1.16 + 2.16) = (2, 1) $$

Check: $2 b_1 + b_2 = (2.9, 1.8)$. When the basis is nearly flat and its arrows are not long, $\det B$ is near $0$ and $B^{-1}$ is large: a small move of $v$ can change its coordinates a lot.

If $b_1$ and $b_2$ have length 1 and meet at a right angle (an orthonormal basis), each coordinate is a dot product, $c_1 = b_1 \cdot v$ and $c_2 = b_2 \cdot v$: the length of $v$'s shadow on that arrow's line, negative if it falls behind the start. Then $B^{-1} = B^\top$, the transpose ($B$ with rows and columns swapped): its rows are $b_1$ and $b_2$, so it takes those dot products.

## The same map in another basis

A map's matrix $A$ is written in the square grid. In the basis $b_1, b_2$ the same map has the matrix

$$ B^{-1}AB $$

Read it from the right: $B$ turns coordinates into an arrow, $A$ moves it, and $B^{-1}$ turns that back into coordinates. Its first column is $Ab_1$ written in steps of $b_1$ and $b_2$. The numbers change, but the determinant does not, since by the product rule $\det(B^{-1}AB) = \det B^{-1} \det A \det B = \det A$. Nor do the eigenvalues, since the map is the same, nor the trace, their sum.

## The eigenvector basis

If $b_1$ and $b_2$ lie along eigenvectors of $A$, then $Ab_1 = \lambda_1 b_1$ and $Ab_2 = \lambda_2 b_2$, with no step along the other. The matrix in that basis is then diagonal, with zeros off its diagonal:

$$ B^{-1}AB = D = \begin{bmatrix} \lambda_1 & 0 \\ 0 & \lambda_2 \end{bmatrix}, \qquad A = BDB^{-1} $$

Example: the map "stretches", with columns $(1.5, 1)$ and $(0.5, 1)$, sends $(1, 1)$ to $(2, 2)$ and $(-1, 2)$ to $(-0.5, 1)$: eigenvalues 2 and 0.5, trace $2.5$ and determinant 1. A diagonal form needs two eigenvector lines: a quarter turn has none, a shear only one.

# In the picture

- The grey grid is the square grid; the blue grid is the grid of $b_1$ and $b_2$, one step of each to a cell.
- The pink arrow is $b_1$, the green $b_2$, the white $v$: drag their tips.
- The wide pink and green bands are $v$'s walk in the basis, $c_1$ steps of $b_1$ then $c_2$ of $b_2$; grey dashes are its walk in the square grid.
- Over the picture, $v = Bc$ and $c = B^{-1}v$ are worked out in numbers.
- With "a map" shown, pale pink and pale green are $Ab_1$ and $Ab_2$, and $A$ and $B^{-1}AB$ are written over the picture.

> Try first "the same arrow, new numbers": $v$ stays where it is while the grid changes under it, and its numbers go from $(2.9, 1.8)$ to $(2, 1)$.

see: span, compose, eigen, svd
