# The idea

A factorisation writes one matrix as a product of simpler ones. Each factor is a map you can picture, so the product plays as a short film of plain motions. This scene takes one $2 \times 2$ matrix $A$ and writes it five ways: LU, QR, with its eigenvectors, the SVD and the polar form. Every film starts with the plane as it is and ends where $A$ takes it. You need matrices as maps of the plane, products of maps, determinants and eigenvectors.

# The mathematics

## Reading a product

In $A = BC$ the factor on the right acts first: $Ax = B(Cx)$. Determinants multiply, $\det A = \det B \det C$, so if $A$ turns the plane over ($\det A < 0$), so must one of its factors. They come in three kinds:

- triangular: upper triangular (zeros below the diagonal) keeps $(1, 0)$ on the $x$ axis, a shear with stretches along the axes; lower triangular keeps $(0, 1)$ on the $y$ axis;
- orthogonal, $Q^\top Q = I$: it keeps lengths and angles, so in the plane it is a turn ($\det Q = 1$) or a mirror in a line ($\det Q = -1$);
- diagonal: a stretch along the $x$ and $y$ axes.

## The five

Take the scene's first matrix,

$$ A = \begin{bmatrix} 1 & 1 \\ 0.5 & 1.5 \end{bmatrix} $$

LU is elimination. Taking $0.5$ times row 1 from row 2 leaves the upper triangular $U$, and the multiplier $0.5$ goes into $L$, which has ones on its diagonal:

$$ A = LU = \begin{bmatrix} 1 & 0 \\ 0.5 & 1 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix} $$

Elimination divides by $a_{11}$. If $a_{11} = 0$ and $a_{21} \ne 0$ there is no LU at all; swapping the rows first gives $PA = LU$.

QR: here $Q$ is the turn that takes $(1, 0)$ onto the direction of the first column $a_1$, here 26.6°, and $R = Q^\top A$ is upper triangular with $r_{11} = |a_1| = 1.12$. If $A$ is invertible and the diagonal of $R$ is kept positive, as in the scene, the two are unique.

Eigenvectors: $A = VDV^{-1}$, where $V$ has eigenvectors as its columns and $D$ the eigenvalues. Here $\lambda^2 - 2.5\lambda + 1 = 0$ gives $\lambda = 2$ and $0.5$, with $A(1, 1) = (2, 2)$ and $A(-2, 1) = (-1, 0.5)$. $V^{-1}$ moves the eigenvectors onto the axes, $D$ stretches along the axes, and $V$ moves them back. It needs two independent eigenvectors, and real numbers need real eigenvalues: a turn (by any angle but 0° or 180°) has no real eigenvalue, and a shear has only one eigenline. One use is powers: $A^k = VD^kV^{-1}$.

The SVD, $A = U\Sigma V^\top$, always exists: $V^\top$ turns, $\Sigma$ stretches along the axes by $\sigma_1 \ge \sigma_2 \ge 0$, and $U$ turns. This scene keeps $V$ a turn, so for $\det A < 0$, $U$ is a mirror instead. (The scene on the SVD keeps $U$ a turn too and lets $\sigma_2$ be negative instead.)

The polar form, $A = QP$, uses the same pieces: $P = V\Sigma V^\top$ stretches along $V$'s columns $v_1$ and $v_2$, at right angles, and $Q = UV^\top$ is a turn, or a mirror if $\det A < 0$. $Q$ is the orthogonal matrix nearest to $A$ (one of them, if $A$ is singular).

# In the picture

- Yellow: the unit circle and the unit square with a lopsided letter, carried by the factors that have acted so far. The faint outline is where $A$ takes them: every film ends on it. Dashed grey is the start.
- Pink and green: $(1, 0)$ and $(0, 1)$ carried along; they end on the columns $a_1$ and $a_2$, the dots you can drag.
- Orange and purple: $v_1$ and $v_2$, in the films that have them.
- On the picture: the factors, the one acting lit, and their product next to $A$, equal but for rounding. A dashed white line is a mirror's line, while the mirror acts.

> Try first "the same map, five films": five different films, one ending.

see: elim, gram, basis, svd
