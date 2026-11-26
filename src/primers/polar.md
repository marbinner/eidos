# The idea

A complex number is a turn times a size, $z = e^{i\theta} r$. Every square matrix can be taken apart the same way, $A = QP$: in the plane, a symmetric $P$ that stretches along two perpendicular axes, then a $Q$ that keeps every length and angle. This is the polar form. Its $Q$ is the turn or mirror nearest to $A$. You need matrices as maps, eigenvectors, and the singular values of the SVD.

# The mathematics

## The stretch $P$

For an arrow $x$, the squared length of $Ax$ is a quadratic form:

$$ |Ax|^2 = x \cdot (A^\top A)\, x $$

$A^\top A$ is symmetric, so it has perpendicular unit eigenvectors $v_1$ and $v_2$; its eigenvalues are squared lengths, $|Av_1|^2 \ge |Av_2|^2$, so not negative: call them $\sigma_1^2 \ge \sigma_2^2$. Take square roots on the same axes:

$$ P = \sqrt{A^\top A}, \qquad P v_1 = \sigma_1 v_1, \quad P v_2 = \sigma_2 v_2 $$

$P$ stretches by $\sigma_1$ along $v_1$ and by $\sigma_2$ along $v_2$, and turns neither axis; $\sigma_1 \ge \sigma_2 \ge 0$ are the singular values of $A$. No other symmetric matrix without negative eigenvalues squares to $A^\top A$, so $A$ fixes $P$.

## The turn $Q$

If $A$ is invertible, so is $P$ ($\sigma_2 > 0$); let $Q = AP^{-1}$. It keeps lengths, because $Q^\top Q = P^{-1} A^\top A P^{-1} = P^{-1} P^2 P^{-1} = I$:

$$ A = QP, \qquad Q^\top Q = I $$

Since $\det A = \det Q \cdot \sigma_1 \sigma_2$, $Q$ is a turn ($\det Q = 1$) when $\det A > 0$ and a mirror in a line ($\det Q = -1$) when $\det A < 0$. Both factors are then unique. If $A$ presses the plane onto a line ($\sigma_2 = 0$), $P$ is still unique, but $Q$ only has to take $v_1$ to $u_1 = Av_1 / \sigma_1$, and a turn and a mirror both do; if $A = 0$, any $Q$ will do.

The other order works too: $A = P'Q$ with $P' = QPQ^\top$, the same stretches along $u_1 = Qv_1$ and $u_2 = Qv_2$. From an SVD $A = U \Sigma V^\top$ the factors are $Q = UV^\top$ and $P = V \Sigma V^\top$.

The symmetric and skew parts, $A = S + K$, are a sum, but near $I$ the two agree to first order: for $A = I + \varepsilon E$ with $\varepsilon$ small, $P \approx I + \varepsilon S_E$ and $Q \approx I + \varepsilon K_E$, the parts of $E$.

## The nearest turn

Let $\|A - B\|$ be the square root of the sum of the squared differences of the entries. Since $Q$ keeps lengths,

$$ \|A - Q\| = \|QP - Q\| = \|P - I\| = \sqrt{(\sigma_1 - 1)^2 + (\sigma_2 - 1)^2} $$

and no orthogonal $O$ is nearer: $\|A - O\|^2 = \|A\|^2 + 2 - 2(\sigma_1 \cos\alpha_1 + \sigma_2 \cos\alpha_2)$, where $O^\top Q$ moves $v_1$ and $v_2$ through the angles $\alpha_1$ and $\alpha_2$, and that is least when both are 0, at $O = Q$. So a turn matrix whose numbers have drifted is mended by taking its $Q$, if $\det A > 0$. If $\det A < 0$, $Q$ is a mirror, and every turn is further off.

## An example

Take

$$ A = \begin{bmatrix} 0 & -2 \\ 1 & 0 \end{bmatrix}, \qquad A^\top A = \begin{bmatrix} 1 & 0 \\ 0 & 4 \end{bmatrix} $$

so $P$ stretches $y$ by 2 ($\sigma_1 = 2$ along $v_1 = (0, 1)$), and $Q = AP^{-1}$ has columns $(0, 1)$ and $(-1, 0)$: the quarter turn. Then $\|A - Q\| = 1 = \|P - I\|$; in the other order $P' = QPQ^\top$ stretches $x$ by 2.

# In the picture

- Yellow: the unit circle and a letter $\text{F}$, carried along; faint, after the first step.
- Orange and purple arrows: $v_1$ and $v_2$, carried along; dashed, the stretch's axes ($v_1$, $v_2$ for $P$; $u_1$, $u_2$ for $P'$).
- Blue: $Q$, the arc of its turn or its mirror line.
- Pink and green dots: the columns of $A$; drag them.
- The chart: $\|A - O\|$ for the turn by $t$ (solid) and the mirror in the line at $t/2$ (dashed); its lowest point is $Q$, at the height $\|P - I\|$ (red).

> Try first "stretch, then turn", then "a mirror inside", where the lowest point moves to the dashed curve.

see: svd, spectral, symskew, factor
