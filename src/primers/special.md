# The idea

Many useful matrices obey one equation, such as $A^\top = A$, which limits what the map can do to the plane and where its eigenvalues can lie. The scene shows eleven such kinds of $2 \times 2$ matrix, and needs matrices as maps of the plane, eigenvalues, and complex numbers $a + bi$ as points $(a, b)$.

# The mathematics

## A kind is an equation

The transpose $A^\top$ is $A$ flipped over its diagonal, $(A^\top)_{ij} = a_{ji}$. It moves $A$ across a dot product:

$$ x \cdot Ay = A^\top x \cdot y $$

Each kind is one equation, three with a condition:

- symmetric $A^\top = A$; skew-symmetric $A^\top = -A$; normal $AA^\top = A^\top A$;
- orthogonal $A^\top A = I$, which says $|Ax| = |x|$ for every $x$;
- projection $A^2 = A$; nilpotent $A^2 = 0$; shear $(A - I)^2 = 0$ with $A \ne I$;
- diagonal: $0$ off the diagonal; upper triangular: $a_{21} = 0$;
- positive definite: symmetric, with $x \cdot Ax > 0$ for every $x \ne 0$;
- Markov: entries $\ge 0$, each column adding to $1$.

## The equation decides the eigenvalues

If $Av = \lambda v$ with $v \ne 0$, then $A^2 v = \lambda^2 v$. So for a projection $\lambda^2 = \lambda$, and $\lambda$ is $0$ or $1$; for a nilpotent matrix $\lambda^2 = 0$; for a shear $(\lambda - 1)^2 = 0$.

For a symmetric matrix with $Av = \lambda v$, $Aw = \mu w$ and $\lambda \ne \mu$,

$$ \lambda\, v \cdot w = Av \cdot w = v \cdot Aw = \mu\, v \cdot w $$

forces $v \cdot w = 0$: the eigenvectors are perpendicular. The eigenvalues are real: for rows $(a, b)$ and $(b, d)$ they are $(a + d)/2 \pm \sqrt{((a - d)/2)^2 + b^2}$, a root of a sum of squares. If it is positive definite too, $\lambda\,|v|^2 = v \cdot Av > 0$, so $\lambda > 0$.

For a skew-symmetric matrix, $x \cdot Ax = A^\top x \cdot x = -Ax \cdot x$ is minus itself, so $0$: $Ax$ is at right angles to $x$. In the plane it is $wJ$, with $J$ the quarter turn, and its eigenvalues are $\pm wi$.

An orthogonal matrix keeps lengths, so a real eigenvalue is $1$ or $-1$ (a reflection has both); a rotation by $\theta$ has $\cos\theta \pm i \sin\theta$, on the unit circle. A triangular matrix's eigenvalues are its diagonal entries, since $\det(A - \lambda I) = (a_{11} - \lambda)(a_{22} - \lambda)$. For a Markov matrix, $(1, 1)A = (1, 1)$: $1$ is an eigenvalue of $A^\top$, so of $A$, and the other, $\tr A - 1$, lies between $-1$ and $1$.

A normal matrix of the plane is symmetric or a turn and a scaling, $aI + bJ$; either way it has eigenvectors at right angles (complex for $aI + bJ$, by the complex dot product).

## Sums, products and inverses

Orthogonal matrices survive products, $(AB)^\top AB = B^\top A^\top A B = I$, and inverses, $A^{-1} = A^\top$; but $I + I = 2I$ doubles lengths. Symmetric ones survive sums and (where there is one) inverses, but $(AB)^\top = B^\top A^\top = BA$: their product is symmetric exactly when $AB = BA$. An example:

$$ A = \begin{bmatrix} 1 & 0.5 \\ 0.5 & -1.5 \end{bmatrix}, \quad B = \begin{bmatrix} 1.5 & -0.5 \\ -0.5 & -0.5 \end{bmatrix}, \quad AB = \begin{bmatrix} 1.25 & -0.75 \\ 1.5 & 0.5 \end{bmatrix} $$

The top left entry is $1 \cdot 1.5 + 0.5 \cdot (-0.5) = 1.25$. The trace of $AB$ is $1.75$ and its determinant $0.625 + 1.125 = 1.75$, so its eigenvalues are $0.875 \pm \sqrt{0.765625 - 1.75} = 0.875 \pm 0.99i$: off the real line, where a symmetric matrix's never are.

# In the picture

- Yellow: the letter and circle after the map; dashed, before.
- Pink and green: where $(1, 0)$ and $(0, 1)$ land; drag the tips.
- Orange and purple, dashed: the eigenlines with their $\lambda$.
- The panel: bright buttons are the kinds the matrix is.
- The complex plane: eigenvalues as dots; bright, where this kind's must lie.
- "Two at once": $B$ blue, $A$ white, the rest about $A + B$, $AB$ or $A^{-1}$.

> Try first "symmetric: right angles": the eigenlines stay at right angles while a column is dragged round.

see: linmap2, eigen, symskew, commute
