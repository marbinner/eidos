# The idea

Many useful matrices pass one test: an equation such as $A^\top = A$, or a condition such as "every entry is above $0$". Each test makes a kind, and the scene has thirty kinds of $2 \times 2$ matrix in seven families. The test limits what the map does to the plane and where its eigenvalues lie, and some kinds lie inside others. You need matrices as maps, the transpose, eigenvalues, and complex numbers $a + bi$ as points $(a, b)$.

# The mathematics

## A kind is an equation

The transpose $A^\top$ is $A$ flipped over its diagonal, $(A^\top)_{ij} = a_{ji}$. Three kinds and their tests:

$$ A^\top = A \quad \text{(symmetric)}, \qquad A^\top A = I \quad \text{(orthogonal)}, \qquad A^2 = A \quad \text{(a projection)} $$

A $2 \times 2$ matrix has 4 numbers; each independent equation fixes one. $A^\top = A$ is one equation, $a_{12} = a_{21}$, so a symmetric matrix has 3 free numbers. $A^\top A$ is always symmetric, so $A^\top A = I$ is three equations, and an orthogonal matrix has 1 free number: an angle. A strict inequality fixes none: matrices with positive entries have all 4 free.

The families group the tests: where the zeros are, symmetry, what the map keeps, what doing it twice does, signs, rank, and names of their own.

## The equation decides the eigenvalues

If $Av = \lambda v$ with $v \ne 0$, then $A^2 v = \lambda^2 v$. So for a projection $\lambda^2 v = \lambda v$, and $\lambda$ is $0$ or $1$. For an involution, $A^2 = I$, $\lambda^2 = 1$. An orthogonal matrix keeps lengths, $|Av| = |v|$, so $|\lambda| = 1$: its eigenvalues lie on the unit circle.

A test made of inequalities works through discs. Each Gershgorin disc is centred on a diagonal entry, with the size of the other entry of its row as radius, and every eigenvalue lies in one of them. If each diagonal entry outweighs the rest of its row (diagonally dominant), $0$ is in neither disc, so $0$ is not an eigenvalue and $A$ has an inverse.

## The map of the zoo

One kind contains another when every matrix of the second passes the first's test. The map draws a line from a kind up to each larger kind just above it; following lines upwards gives every containment: a rotation is orthogonal, an orthogonal matrix normal. A containment is proved by algebra. A missing one is shown by a single matrix: the quarter turn

$$ J = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} $$

is orthogonal but not symmetric, so the map has no line from orthogonal to symmetric.

Example: the slanted mirror

$$ A = \begin{bmatrix} 1 & 1 \\ 0 & -1 \end{bmatrix}, \qquad A^2 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} $$

is an involution. It is not orthogonal: $A$ sends $(0, 1)$ to $(1, -1)$, of length $\sqrt{2}$. So involutions do not lie inside the orthogonal kind, though reflections lie in both.

## Sums and products

Orthogonal matrices survive products, since $(AB)^\top AB = B^\top A^\top A B = I$. For symmetric $A$ and $B$, $(AB)^\top = B^\top A^\top = BA$, so $AB$ is symmetric exactly when $AB = BA$.

Example: write $J$ for the quarter turn, with $J^2 = -I$. Then $aI + bJ$ multiplies like the complex number $a + bi$, because $i^2 = -1$ too. With $A = I + 0.5J$ and $B = 1.2I + 0.9J$,

$$ (1 + 0.5i)(1.2 + 0.9i) = 1.2 - 0.45 + (0.9 + 0.6)i = 0.75 + 1.5i $$

so $AB = 0.75I + 1.5J$, a scaled rotation, equal to $BA$.

# In the picture

- Yellow: a letter and the unit circle after the map; dashed, before it.
- Pink and green: where $(1, 0)$ and $(0, 1)$ land, the columns of $A$. Dragging a tip keeps $A$ of its kind.
- Orange and purple, dashed: the eigenlines, each with its $\lambda$.
- The complex plane: the eigenvalues as dots, and in bright, where this kind's can lie; cyan circles, the Gershgorin discs.
- The tabs: the families, with a light for each kind $A$ is.
- The map: the kinds, the lines between them, and in bright, every kind $A$ is.

> Try first "one matrix, many kinds": the swap of the axes lights eleven kinds at once.

see: linmap2, eigen, symskew, commute
