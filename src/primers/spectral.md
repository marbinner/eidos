# The idea

A matrix is symmetric when it is its own mirror image across the diagonal, $A^\top = A$: the entry in row 1, column 2 equals the one in row 2, column 1. They are everywhere: quadratic forms, covariance matrices, second derivatives (if continuous). The spectral theorem says what every one does: it stretches along axes at right angles, its eigenvectors, each by its eigenvalue. You need eigenvalues and eigenvectors, the dot product, and the transpose.

# The mathematics

## Why the axes are at right angles

For a symmetric $A$, the matrix can be moved across a dot product:

$$ (Av) \cdot w = v^\top A^\top w = v^\top A w = v \cdot (Aw) $$

If $Av = \lambda_1 v$ and $Aw = \lambda_2 w$, this says $\lambda_1 (v \cdot w) = \lambda_2 (v \cdot w)$: if the eigenvalues differ, $v \cdot w = 0$. The eigenvalues are real, too. For a $2 \times 2$ matrix with rows $(a, b)$ and $(b, c)$ they are

$$ \lambda = \frac{a + c}{2} \pm \sqrt{\left(\frac{a - c}{2}\right)^2 + b^2} $$

and the number under the root is a sum of squares, never negative. It is 0 only when $a = c$ and $b = 0$, that is $A = \lambda I$; then every direction is an eigenvector, and any two at right angles will do.

## A sum of projections

Take unit eigenvectors $q_1$, $q_2$ at right angles. Every $x$ splits into its shadows on the two axes, $x = (q_1 \cdot x)\, q_1 + (q_2 \cdot x)\, q_2$. The matrix $P_i = q_i q_i^\top$ casts the shadow on axis $i$: $P_i x = (q_i \cdot x)\, q_i$. Apply $A$ to each shadow, which only multiplies it by its eigenvalue, and add:

$$ A = \lambda_1 q_1 q_1^\top + \lambda_2 q_2 q_2^\top = Q \Lambda Q^\top $$

Here $Q$ has columns $q_1$ and $q_2$, so $Q^\top Q = I$, and $\Lambda$ has $\lambda_1$, $\lambda_2$ on its diagonal.

Example, the scene's first: $q_1 = (0.8, 0.6)$, $q_2 = (-0.6, 0.8)$, $\lambda_1 = 2$, $\lambda_2 = 0.5$ give

$$ A = 2 \begin{bmatrix} 0.64 & 0.48 \\ 0.48 & 0.36 \end{bmatrix} + 0.5 \begin{bmatrix} 0.36 & -0.48 \\ -0.48 & 0.64 \end{bmatrix} = \begin{bmatrix} 1.46 & 0.72 \\ 0.72 & 1.04 \end{bmatrix} $$

For $x = (0.4, 1.55)$, $q_1 \cdot x = 1.25$ and $q_2 \cdot x = 1$, so $Ax = 2.5\, q_1 + 0.5\, q_2 = (1.7, 1.9)$. Directly: $1.46 \cdot 0.4 + 0.72 \cdot 1.55 = 1.7$ and $0.72 \cdot 0.4 + 1.04 \cdot 1.55 = 1.9$.

## What the eigenvalues do

The unit circle becomes an ellipse with half-axes $|\lambda_1|$ and $|\lambda_2|$ along the axes. A negative eigenvalue also turns its axis round: if just one is negative, the plane is flipped over, and $\det A = \lambda_1 \lambda_2 < 0$. An eigenvalue $\lambda_2 = 0$ squashes the plane onto the first axis, $A = \lambda_1 P_1$, and leaves no inverse.

Since $P_1^2 = P_1$, $P_2^2 = P_2$ and $P_1 P_2 = 0$, powers act on the eigenvalues alone: $A^2 = \lambda_1^2 P_1 + \lambda_2^2 P_2$. So do other functions, on the same axes: $A^{-1}$ has $1/\lambda$ (if no eigenvalue is 0), and the square root that keeps the axes has $\sqrt{\lambda}$ (if none is negative).

## Without the symmetry

Add a skew part: $A = S + kJ$, where $S$ is symmetric with eigenvalues $\lambda_1$, $\lambda_2$ and $J$ turns every arrow a quarter turn. The eigenvalues of $A$ are real only while $|k| \le |\lambda_1 - \lambda_2|/2$. Below that, its two eigenlines are $\arccos(2|k|/|\lambda_1 - \lambda_2|)$ apart: no longer at right angles, so the shadows must be cast along the other line. At $|k| = |\lambda_1 - \lambda_2|/2$ the lines meet; beyond, the eigenvalues are complex and no line through $0$ is kept.

# In the picture

- Orange and purple dashed lines: the eigenlines. Their dots, $|\lambda_1|$ and $|\lambda_2|$ from $0$, can be dragged.
- Yellow, dashed: the unit circle and a letter; solid: where the map takes them.
- White: $x$ and $Ax$; dashed white: $x$ dropped onto the axes. Faint arrows: $P_1 x$, $P_2 x$; solid: $\lambda_1 P_1 x$, $\lambda_2 P_2 x$, added by the dashed parallelogram.
- The panel on the picture: $A$ and $\lambda_1 P_1 + \lambda_2 P_2$, with both projections written out.

> Try first "a sum of two projections": watch $x$ split, stretch and add up to $Ax$.

see: eigen, quadform, project, svd
