# The idea

A linear map moves every point of the plane at once, keeping $0$ in place and the lines of a square grid straight, parallel and evenly spaced. It is settled by where it sends two arrows, and those two landing places, side by side, are its matrix: a $2 \times 2$ table of numbers. This scene lets you set the matrix, watch the plane move, and see what the map does to areas, which lines stay put, and how much it stretches. You need the span primer: walks along arrows.

# The mathematics

## The columns say where everything goes

Write $\hat\imath = (1, 0)$ and $\hat\jmath = (0, 1)$. Every point is a walk along them, $(x, y) = x\hat\imath + y\hat\jmath$. A linear map $A$ keeps walks as walks: it sends a sum of arrows to the sum of where they land, and $c$ times an arrow to $c$ times where it lands. So where $\hat\imath$ and $\hat\jmath$ land settles where every point lands (the point $(x, y)$ is written as a column):

$$ A \begin{bmatrix} x \\ y \end{bmatrix} = x \, \textcolor{pink}{A\hat\imath} + y \, \textcolor{green}{A\hat\jmath} $$

The matrix of $A$ has $A\hat\imath$ as its first column and $A\hat\jmath$ as its second. Entry by entry,

$$ A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}, \qquad A \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} ax + by \\ cx + dy \end{bmatrix} $$

Example: the scene opens with $A\hat\imath = (1.5, 0.5)$, $A\hat\jmath = (-0.5, 1.2)$ and the white dot at $v = (1, 1.5)$, so

$$ Av = 1 \times (1.5,\ 0.5) + 1.5 \times (-0.5,\ 1.2) = (0.75,\ 2.3) $$

## Areas: the determinant

The unit square, with sides $\hat\imath$ and $\hat\jmath$, lands on the parallelogram with sides $A\hat\imath$ and $A\hat\jmath$. Its area is the determinant, counted negative when the plane has been flipped over:

$$ \det A = ad - bc $$

Every small square of a fine grid lands on a scaled copy of it, so every area is multiplied by $|\det A|$. Here $\det A = 1.5 \times 1.2 - (-0.5) \times 0.5 = 2.05$.

## Lines that stay put: eigenvectors

An eigenvector is an arrow $v$, not $0$, that stays on its own line and is only stretched:

$$ Av = \lambda v $$

The number $\lambda$ is its eigenvalue; a negative $\lambda$ reverses the arrow, and $\lambda = 0$ sends it to $0$. Example: the reflection that swaps $x$ and $y$ sends $(1, 1)$ to itself ($\lambda = 1$) and $(1, -1)$ to $(-1, 1)$, which is $-1$ times it ($\lambda = -1$). A turn by 45° moves every line off itself, so it has no eigenvector lines; its eigenvalues are complex numbers, not real ones. So are those of the matrix the scene opens with: $1.35 \pm 0.477i$. The eigenvalues add up to the trace $\tr A = a + d$ and multiply to $\det A$.

## Stretches: singular values, and rank

The unit circle, every arrow of length 1, lands on an ellipse. Its longest half-axis $\sigma_1$ is the most the map stretches any arrow, its shortest $\sigma_2$ the least: these are the singular values. The ellipse has area $\pi\sigma_1\sigma_2$ and the circle $\pi$, so $\sigma_1 \sigma_2 = |\det A|$; here $1.590 \times 1.290 \approx 2.05$.

When $\det A = 0$ the columns are dependent: the plane is pressed onto a line (or onto $0$), the ellipse is flat and $\sigma_2 = 0$. The rank of $A$ is the rank of its columns, the dimension of where the plane lands: 2, 1 or 0.

# In the picture

- The pink arrow, labelled $\hat\imath$, is $A\hat\imath$, the first column; the green one is $A\hat\jmath$, the second. Drag their tips, or type the numbers.
- The blue grid and the yellow parallelogram are the square grid and the unit square after the map; the yellow turns pink when $\det A$ is negative.
- The white dot is $v$, the white arrow $Av$.
- Dashed orange and purple lines are the eigenvector lines, each with an arrow stretched by its $\lambda$. There are none at the start.
- With the circle shown, the white ellipse is where the unit circle lands, and its solid orange and purple half-axes are $\sigma_1$ and $\sigma_2$.

> Try first "a shear": the square leans over but keeps its area, and only the $x$ axis stays on itself.

see: span, linmap3, compose, det
