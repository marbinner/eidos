# The idea

An $n \times n$ matrix is $n^2$ numbers, but the large matrices met in practice are rarely arbitrary. They have a pattern: zeros in fixed places, entries that repeat, rows or columns with a fixed sum. The pattern pays three ways: fewer numbers to store, less work to solve $Ax = b$, and a known region where the eigenvalues must lie. It builds on the product $Ax$, eigenvalues (complex ones too) and singular values.

# The mathematics

## Zeros save work

Elimination clears the entries below the diagonal column by column, then solves from the bottom up: for a full matrix,

$$ \tfrac{1}{3}n^3 + n^2 - \tfrac{1}{3}n $$

multiplications and divisions: 106 at $n = 6$. If every entry more than $p$ places below the diagonal or $q$ above it is 0, elimination without row exchanges never fills in outside that band: a tridiagonal matrix takes $5n - 4$, a triangular one $n(n+1)/2$, a diagonal one $n$ divisions.

Some patterns turn solving into something else. The difference matrix $D$ has 1 on the diagonal and $-1$ just below it, so $(Dx)_i = x_i - x_{i-1}$. Solving $Dx = b$ from the top gives $x_1 = b_1$, $x_2 = b_1 + b_2$, and so on: running sums. With $b = (1, 0.5, -1)$, $x = (1, 1.5, 0.5)$; check: $Dx = (1, 1.5 - 1, 0.5 - 1.5)$. So $D^{-1}$ is the lower triangle of ones.

## Patterns place the eigenvalues

For a triangular matrix, $\det(A - \lambda I)$ is the product of the $a_{ii} - \lambda$, so the eigenvalues are the diagonal entries. For any matrix, if $Ax = \lambda x$, take the entry $x_i$ of largest size: row $i$ says $(\lambda - a_{ii})x_i = \sum_{j \ne i} a_{ij}x_j$, so

$$ |\lambda - a_{ii}| \le \sum_{j \ne i} |a_{ij}| $$

Every eigenvalue lies in one of these Gershgorin discs. If each diagonal entry outweighs the rest of its row, no disc reaches 0, and the matrix has an inverse. Other patterns say more:

- Circulant (each row the one above moved one place along, wrapping round): with $\omega = e^{2\pi i/n}$, $(1, \omega^k, \omega^{2k}, \dots)$ is an eigenvector of every circulant, and $\lambda_k = \sum_j c_j \omega^{jk}$, the Fourier transform of the first row $c$.
- Markov (no negative entries, columns adding to 1): the row of ones times $M$ is the row of ones, so 1 is an eigenvalue; the discs of $M^\top$, which has the same eigenvalues, never reach outside the unit circle.
- Graph Laplacian $L$: $x^\top L x$ is the sum of $(x_i - x_j)^2$ over the edges, never negative, and $Lx = 0$ exactly when $x$ is constant on each piece: one zero eigenvalue for each piece.
- Permutation: on a cycle of length $\ell$ the entries go round a ring, giving the $\ell$th roots of 1.
- Companion matrix $C$ of $p(\lambda) = \lambda^n + c_{n-1}\lambda^{n-1} + \dots + c_0$: the row $(1, \lambda, \dots, \lambda^{n-1})$ times $C$ is $\lambda$ times itself exactly when $p(\lambda) = 0$, so the eigenvalues are the roots of $p$.

## How sensitive: singular values

The condition number $\kappa = \sigma_1/\sigma_n$ bounds how much a change in $b$ can be magnified in the answer:

$$ \frac{|\Delta x|}{|x|} \le \kappa \, \frac{|\Delta b|}{|b|} $$

The Hilbert matrix, $h_{ij} = 1/(i + j - 1)$, has only positive eigenvalues, yet at $n = 6$ its $\kappa$ is $1.5 \times 10^7$. A matrix with no inverse has $\sigma_n = 0$ and $\kappa = \infty$.

# In the picture

- The grid is the matrix: teal above 0, rose below. An empty cell is a 0 the pattern forces; an inner frame, an entry it sets.
- White bars are what you drag ($x$, or $b$ when solving); yellow bars are $Ax$, or the solution.
- In the plane, orange dots are the eigenvalues, green is the kind's promise, blue circles the Gershgorin discs.
- Cyan dots are the singular values on a scale of powers of ten, with $\kappa$.

> Try first "cut an edge of the graph": an eigenvalue slides to 0 as the graph falls into two pieces.

see: special, schur, dft, condition
