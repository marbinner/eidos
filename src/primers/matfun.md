# The idea

For a number $a > 0$, $a^t = e^{t \ln a}$ makes sense for every $t$. This scene asks what $A^{1/2}$ or $A^t$ can mean for a $2 \times 2$ matrix $A$. The answer comes through the eigenvalues, or through a logarithm $L$, with $e^L = A$: then $A^t$ is a path of maps from $I$ to $A$, and halfway along it is a square root $B$, with $B^2 = A$. Some matrices have many of each, and some none. You need eigenvalues and eigenvectors, and the matrix exponential $e^{At}$.

# The mathematics

## Through the eigenvalues

If $A$ has a basis of eigenvectors, the columns of $V$, then $A = VDV^{-1}$ with the eigenvalues in the diagonal matrix $D$. Then $A^2 = VD^2V^{-1}$, and the same for every power, so a function given by a power series acts on each eigenvalue alone:

$$ f(A) = V f(D) V^{-1} $$

The recipe works for any $f$ defined at the eigenvalues. For the square root, $B = VD^{1/2}V^{-1}$ has $B^2 = VDV^{-1} = A$. The scene's stretch

$$ A = \begin{bmatrix} 1.04 & 0.4 \\ 0.4 & 1.04 \end{bmatrix} $$

has $\lambda = 1.44$ along $(1, 1)$ and $0.64$ along $(1, -1)$. Their square roots are $\pm 1.2$ and $\pm 0.8$: four choices of sign, four square roots. With both signs positive,

$$ B = \begin{bmatrix} 1 & 0.2 \\ 0.2 & 1 \end{bmatrix} $$

(check: $1 + 0.04 = 1.04$ and $0.2 + 0.2 = 0.4$). There are no others: when $A$ has two different eigenvalues, any $B$ with $B^2 = A$ keeps $A$'s eigenvectors, since $B$ and $A = B^2$ commute.

## Logarithms, and $A^t$

If $e^L = A$, define

$$ A^t = e^{tL} $$

It is $I$ at $t = 0$, $A$ at $t = 1$, $A^2$ at $t = 2$, and $A^s A^t = A^{s+t}$: $A^{1/2}$ done twice is $A$. Its determinant is $e^{t \tr L} = (\det A)^t$, never 0.

A turn by the angle $\theta$ has the logarithm $\theta J$, where $J$ is the quarter turn, because $e^{\theta J} = \cos\theta\, I + \sin\theta\, J$. But $\theta$ and $\theta + 360^\circ$ give the same turn, so logarithms are not unique: for the turn by 170°, $170^\circ \times J$ and $-190^\circ \times J$ ($2.97J$ and $-3.32J$ in radians) both work. The two paths go round opposite ways, and halfway they are the turns by 85° and −95°: the two real square roots. The principal logarithm keeps the angle within half a turn; it exists when no eigenvalue of $A$ is 0 or negative.

## When there is none

A real square root needs $\det A \ge 0$, and a real logarithm $\det A > 0$, as

$$ \det B^2 = (\det B)^2 \ge 0, \qquad \det e^L = e^{\tr L} > 0 $$

So a mirror, with $\det A = -1$, has neither, and no real $A^t$. A positive determinant is not enough: a matrix with $\lambda = -1$ and $-2$ has no real square root. Nor is an eigenvalue at $-1$ always fatal: $-I = e^{\pi J}$, the half turn.

The straight blend $(1 - t)I + tA$ is no substitute: for the turn by 170° at $t = 0.5$ its determinant is $(1 + \cos 170^\circ)/2 = 0.0076$. For the mirror it is flat at $t = 0.5$; any path from $I$ to a mirror without jumps is flat somewhere, as its determinant goes from 1 to $-1$.

A shear $I + N$, with $N^2 = 0$, has no basis of eigenvectors, but the series still works: $\log(I + N) = N$, and $A^t = I + tN$.

# In the picture

- Yellow: the letter and the unit circle carried by $A^t$, or by $B^2 = A$; dashed, where they start.
- Blue: the straight blend; with the square roots, $B$ applied once (the arrows marked $B$ move the corner).
- Pink, green and white: the paths of $(1, 0)$, $(0, 1)$ and the corner, $t$ from 0 to 2; dashed white, the principal path when another logarithm is chosen.
- Orange and purple: the eigenlines; in the complex plane, the eigenvalues $\lambda^t$, rings at $\lambda$.

> Try first "the straight blend collapses": the yellow letter turns, while the blue one shrinks almost to nothing.

see: expm, basis, powers, jordan
