# The idea

A matrix is easiest to understand in a basis of its eigenvectors, where it only stretches; a $2 \times 2$ matrix needs two of them. With an eigenvalue repeated it may have only one eigenline, and is then called defective. Here is the best a change of basis can do for it, what its powers do, and why its eigenvalues move easily. You need eigenvalues, eigenvectors and a change of basis.

# The mathematics

## One eigenline

The two eigenvalues are equal, both $\lambda = \tr A / 2$, when $(\tr A)^2 = 4 \det A$. Then $N = A - \lambda I$ squares to $0$: either $N = 0$, and $A = \lambda I$ keeps every line, or $N$ flattens the plane onto one line and sends that line to $0$, the single eigenline.

## The Jordan block

Take any $w$ off the eigenline. Then $v = Nw$ is not $0$ and lies on the eigenline, so

$$ Av = \lambda v, \qquad Aw = \lambda w + v $$

With $P = [\,v\;\; w\,]$, whose columns are $v$ and $w$, these say $AP = PJ$:

$$ A = PJP^{-1}, \qquad J = \begin{bmatrix} \lambda & 1 \\ 0 & \lambda \end{bmatrix} $$

No basis makes it diagonal: a diagonal matrix with $\lambda$ twice would be $\lambda I$. In a larger matrix a chain can be longer, and its block has a 1 in every place just above the diagonal.

## Powers: a shear that keeps growing

$J = \lambda I + N$ with $N^2 = 0$, so multiplying out $(\lambda I + N)^k$ leaves only two terms:

$$ J^k = \begin{bmatrix} \lambda^k & k\lambda^{k-1} \\ 0 & \lambda^k \end{bmatrix} $$

So if $x = c_1 v + c_2 w$, then $A^k x = (c_1 \lambda^k + c_2 k\lambda^{k-1})\, v + c_2 \lambda^k w$; the flow is alike, $e^{Jt} = e^{\lambda t}(I + tN)$. When $|\lambda| < 1$ everything shrinks to $0$ in the end, but $k|\lambda|^{k-1}$ can grow first, while $(k + 1)|\lambda| / k$, the factor from one step to the next, is above 1: for $\lambda = 0.9$, up to $k = 9$.

Example, the scene's opening: $\lambda = 0.9$, $v$ and $w$ of length 1 at right angles, and $x = 1.25w$. At $k = 9$, $k\lambda^{k-1} = 9 \cdot 0.9^8 = 3.874$ and $\lambda^9 = 0.387$, so $|A^9 x| = 1.25\sqrt{3.874^2 + 0.387^2} = 4.87$: 3.89 times $|x|$, its highest, while $0.9^k$ only shrinks.

## How it arises, and why it is delicate

Put a small $\varepsilon$ in place of the 0 below $J$'s diagonal. The eigenvalues of this $J_\varepsilon$ solve $(\mu - \lambda)^2 = \varepsilon$:

$$ \mu = \lambda \pm \sqrt{\varepsilon} $$

For $\varepsilon > 0$ there are two eigenlines, along $v \pm \sqrt{\varepsilon}\, w$. As $\varepsilon$ shrinks they close like scissors onto $v$, and the matrix $V$ of eigenvectors comes ever nearer to flat; at $\varepsilon = 0$ its columns lie on one line. For $\varepsilon < 0$ the eigenvalues are $\lambda \pm i\sqrt{-\varepsilon}$, and in a suitable basis it is a turn with a stretch. Up to a change of basis every real $2 \times 2$ matrix is one of these three kinds: diagonal, a Jordan block, a turn with a stretch.

The square root makes a Jordan block delicate: changing that entry by $\varepsilon = 0.0001$ moves the eigenvalues by $0.01$, a hundred times as far. Change a symmetric matrix by a symmetric $E$ whose squared entries add to $0.0001^2$, and its eigenvalues move by at most $0.0001$.

# In the picture

- Orange, dashed: the eigenline; purple: the second one, when $\varepsilon > 0$.
- Pink and green: $v$ and $w$, the columns of $P$ (drag their tips); the faint grid steps along both.
- White: $x$ (drag it) and its steps $x_k = A^k x$, with $x_k$'s parts along $v$ and $w$. Yellow: the unit eigenvectors' parallelogram, of area $|\det V|$.
- The panel: $A = PJP^{-1}$, $J^k$, bars of $|A^k x|$, the eigenvalues, and how far they move against $\varepsilon$.

> Try first "it grows before it shrinks": the steps run out beside the eigenline and come back along it, and the bars climb to $k = 9$ while the orange line only falls.

see: eigen, basis, powers, expm
