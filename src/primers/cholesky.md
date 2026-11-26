# The idea

A positive definite matrix $A$, as covariance matrices usually are, can be written as $A = LL^\top$ with $L$ lower triangular: only zeros above its diagonal. $L$ comes out of three short steps, elimination in another form, and it turns independent random numbers into correlated ones. You need matrix multiplication, the transpose, elimination's pivots, and what a variance is.

# The mathematics

## Positive definite

A symmetric matrix $A$ is positive definite when $x^\top A x > 0$ for every $x$ other than $0$. For

$$ A = \begin{bmatrix} a_{11} & a_{21} \\ a_{21} & a_{22} \end{bmatrix} $$

that comes to $a_{11} > 0$ and $\det A > 0$. Two numbers with standard deviations $\sigma_1$, $\sigma_2$ and correlation $r$ have the covariance matrix

$$ A = \begin{bmatrix} \sigma_1^2 & r\sigma_1\sigma_2 \\ r\sigma_1\sigma_2 & \sigma_2^2 \end{bmatrix} $$

whose determinant is $\sigma_1^2 \sigma_2^2 (1 - r^2)$: it is positive definite exactly when $r$ lies strictly between $-1$ and $1$.

## The recipe, and why it can fail

Multiply out $LL^\top$ for a lower triangular $L$:

$$ LL^\top = \begin{bmatrix} l_{11}^2 & l_{11} l_{21} \\ l_{11} l_{21} & l_{21}^2 + l_{22}^2 \end{bmatrix} $$

Matching it with $A$ gives three steps, each using the one before:

$$ l_{11} = \sqrt{a_{11}}, \qquad l_{21} = a_{21}/l_{11}, \qquad l_{22} = \sqrt{a_{22} - l_{21}^2} $$

With positive roots, this is the only lower triangular $L$ with a positive diagonal and $LL^\top = A$.

Why a root fails, or gives $0$, exactly when $A$ is not positive definite: if $a_{11} \le 0$, then $x = (1, 0)$ gives $x^\top A x = a_{11} \le 0$. If $a_{11} > 0$, let $m = a_{21}/a_{11}$ and complete the square:

$$ x^\top A x = a_{11}(x_1 + m x_2)^2 + (a_{22} - l_{21}^2)\, x_2^2 $$

The numbers in front are those under the roots. If both are positive, so is $x^\top A x$ for every $x \ne 0$. If the second is not, $x = (-m, 1)$ makes $x^\top A x$ equal to it. When it is exactly 0, $A$ is semidefinite: $l_{22} = 0$, and $L$ squashes the plane onto a line.

## Elimination, and other square roots

Elimination takes $m$ times row 1 from row 2, leaving the pivots $a_{11} = l_{11}^2$ and $a_{22} - l_{21}^2 = l_{22}^2$; and $m = l_{21}/l_{11}$. So $L$ stretches the plane along the axes, by $l_{11}$ and $l_{22}$, then shears it, raising each point by $m$ times its first number. For an $n \times n$ matrix this is half the work of finding $L$ and $U$: about $n^3/3$ arithmetic operations against $2n^3/3$.

Any $B = LQ$ with $Q$ orthogonal ($Q^\top Q = I$: a turn or a mirror) also has $BB^\top = LQQ^\top L^\top = A$, and for a positive definite $A$ there are no others. One of them, $S$, is symmetric with no negative eigenvalue: it has the eigenvectors of $A$ and the square roots of its eigenvalues.

## Correlated numbers from independent ones

Let $z = (z_1, z_2)$ be two independent standard normal numbers, whose covariance matrix is $I$. Then $x = Lz$ has covariance $LIL^\top = A$. Written out,

$$ x_1 = l_{11} z_1, \qquad x_2 = l_{21} z_1 + l_{22} z_2 $$

and the $z_1$ they share correlates them. The circle $z^\top z = 1$ goes to the ellipse $x^\top A^{-1} x = 1$, since $A^{-1} = (L^\top)^{-1} L^{-1}$ turns $x^\top A^{-1} x$ back into $z^\top z$.

Example: $\sigma_1 = \sigma_2 = 1$ and $r = 0.8$ give $l_{11} = 1$, $l_{21} = 0.8$ and $l_{22} = \sqrt{1 - 0.64} = 0.6$. So $x_1 = z_1$ and $x_2 = 0.8 z_1 + 0.6 z_2$, each with variance 1 ($0.64 + 0.36$), and covariance $0.8$. With $r = 1.2$ instead, step 3 asks for $\sqrt{1 - 1.44} = \sqrt{-0.44}$, and $x = (-1.2, 1)$ gives $x^\top A x = 1.44 - 2.88 + 1 = -0.44$.

# In the picture

- Left: 400 pairs $z$, the unit circle (yellow), $(1, 0)$ and $(0, 1)$ (pink, green). Right: the pairs as $x$, the ellipse, and $L$'s columns: $(l_{11}, l_{21})$ at the ellipse's rightmost point, $(0, l_{22})$ straight up.
- The dashed box reaches $\sigma_1$ across and $\sigma_2$ up: drag its corner, or where the ellipse touches its side.
- Below: $A = LL^\top$ in numbers, and the three steps. The switch to $S$ reaches the same ellipse another way.

> Try first "correlation from independence": the round cloud is squeezed, then sheared into a correlated one.

see: elim, quadform, gauss2, spectral
