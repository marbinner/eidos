# The idea

An invertible matrix $A$ answers $Ax = b$ exactly once for every $b$. One that squashes the plane onto a line has no inverse: most $b$ are reached by no $x$, and those on the line by infinitely many. The pseudoinverse $A^+$ gives an answer anyway: come as near to $b$ as $A$ can, and of the $x$ that do, take the shortest. It is built from the SVD (the singular value decomposition).

# The mathematics

## The SVD, turned round

Write $A = U \Sigma V^\top$: $A$ takes $v_1$ to $\sigma_1 u_1$ and $v_2$ to $\sigma_2 u_2$, with $\sigma_1 \ge \sigma_2 \ge 0$, $v_1 \perp v_2$ and $u_1 \perp u_2$. If both $\sigma$ are above $0$, undoing $A$ takes the part of $b$ along each $u$ back to its $v$, divided by its $\sigma$:

$$ A^{-1} b = \frac{u_1 \cdot b}{\sigma_1}\, v_1 + \frac{u_2 \cdot b}{\sigma_2}\, v_2 $$

If $\sigma_2 = 0$, the second term cannot be had: no $Ax$ has any part along $u_2$. The pseudoinverse leaves it out. Every $\sigma$ that is not $0$ is turned upside down, every $0$ stays $0$:

$$ A^+ = V \Sigma^+ U^\top, \qquad \Sigma^+ = \begin{bmatrix} 1/\sigma_1 & 0 \\ 0 & 0 \end{bmatrix} \quad \text{when } \sigma_2 = 0 < \sigma_1 $$

With an inverse, nothing is left out: $A^+ = A^{-1}$.

## Nearest, then shortest

With $\sigma_2 = 0 < \sigma_1$, $x^+ = A^+ b = (u_1 \cdot b / \sigma_1)\, v_1$, and $A$ takes it to $p = (u_1 \cdot b)\, u_1$, the shadow of $b$ on the column space (the line of $u_1$). The left-over $b - p = (u_2 \cdot b)\, u_2$ is at right angles to that line, so by Pythagoras, for every $x$,

$$ |Ax - b|^2 = |Ax - p|^2 + |p - b|^2 $$

No $Ax$ is nearer to $b$ than $p$, so $x^+$ is a least-squares answer. The other $x$ with $Ax = p$ are $x^+ + t\, v_2$, as $A$ sends the null space, the line of $v_2$, to $0$. As $x^+$ is along $v_1$, at right angles to $v_2$, their length is $\sqrt{|x^+|^2 + t^2}$: $x^+$ is the shortest. So $AA^+$ projects onto the column space, and $A^+A$ onto the row space, the line of $v_1$.

Example: $A = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}$ has $\sigma_1 = 2$, $\sigma_2 = 0$ and $u_1 = v_1 = (1, 1)/\sqrt{2}$, so $A^+ = \frac{1}{4} \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}$. For $b = (3, 1)$: $x^+ = (1, 1)$, $p = Ax^+ = (2, 2)$, and the left-over $(1, -1)$ is at right angles to $(1, 1)$. Every $(1 + t, 1 - t)$ also gives $(2, 2)$; its squared length, $2 + 2t^2$, is least at $t = 0$.

## A small $\sigma$: cut it off, or damp it

When $\sigma_2$ is small but not $0$, $A^+ = A^{-1}$ turns the part of $b$ along $u_2$ into a part along $v_2$ that is $1/\sigma_2$ times as large. With $\sigma_2 = 0.03$, an error of $0.01$ along $u_2$ in a measured $b$ becomes $0.33$ in $x$. Two remedies:

- cut off: treat every $\sigma$ below a chosen $\tau$ as $0$;
- damp: replace each $1/\sigma$ by $\sigma / (\sigma^2 + \lambda)$, near $1/\sigma$ when $\sigma^2$ is much larger than $\lambda$ and never above $1/(2\sqrt{\lambda})$. This is ridge regression, $(A^\top A + \lambda I)^{-1} A^\top b$.

Noise along $u_2$ then moves the answer less. The price: in general $Ax$ is no longer $b$, even without noise.

A matrix of any shape has $A^+ = V \Sigma^+ U^\top$: with independent columns it is $(A^\top A)^{-1} A^\top$, with independent rows $A^\top (A A^\top)^{-1}$.

# In the picture

- Yellow: the unit circle (inputs) and the ellipse $A$ makes of it (outputs), with half-axes $\sigma_1 u_1$ (orange) and $\sigma_2 u_2$ (purple); drag their ends.
- With $\sigma_2 = 0$: row space and column space (orange), null space (purple, among the inputs).
- White: $b$, to drag. Blue: $x^+$ and $p$; dashed, every $x$ that also reaches $p$. Red: $b - p$.
- Cyan, green: the cut-off and damped answers. The chart: what each $\sigma$ becomes.

> Try first "no inverse: the nearest, then the shortest", then "nearly flat: the inverse runs away".

see: svd, lsq, underdet, condition
