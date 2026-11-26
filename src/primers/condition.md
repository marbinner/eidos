# The idea

Two equations in two unknowns, $Ax = b$, are two lines, and the answer $x$ is where they cross. But $b$ is usually measured, and every measurement is a little off. How far can that move the answer, as a share of its size? One number of the matrix settles it, its condition number $\kappa$: the answer's relative error is at most $\kappa$ times that of $b$, and nearly parallel lines make $\kappa$ large. You need $2 \times 2$ matrices as maps, the inverse $A^{-1}$ that undoes $A$, and lengths.

# The mathematics

## The error in the answer

Say the reading is off by $\delta b$, an arrow of length $\varepsilon$. The answer moves to $x + \delta x$, and since $A(x + \delta x) = b + \delta b$ and $Ax = b$,

$$ A\,\delta x = \delta b, \qquad \delta x = A^{-1} \delta b $$

So the error in $x$ depends only on $A$ and the error in $b$. As $\delta b$ goes round a circle of radius $\varepsilon$, $\delta x$ goes round an ellipse. The relative errors are $|\delta b| / |b|$ and $|\delta x| / |x|$.

## Most and least stretch

Of all arrows of length 1, $A$ stretches one, $v_1$, the most, by $\sigma_1$, and the one at right angles to it, $v_2$, the least, by $\sigma_2$. These are the singular values; their images are at right angles too:

$$ A v_1 = \sigma_1 u_1, \qquad A v_2 = \sigma_2 u_2 $$

with $u_1$ and $u_2$ of length 1. So $A^{-1}$ divides an error along $u_1$ by $\sigma_1$, and one along $u_2$ by $\sigma_2$: the ellipse's half-axes are $\varepsilon / \sigma_1$ and $\varepsilon / \sigma_2$.

## The condition number

$$ \kappa = \sigma_1 / \sigma_2 $$

Why: $A$ stretches every arrow by at least $\sigma_2$ and at most $\sigma_1$, so $|\delta b| = |A\,\delta x| \ge \sigma_2 |\delta x|$ and $|b| = |Ax| \le \sigma_1 |x|$. Together,

$$ \frac{|\delta x|}{|x|} \le \kappa \, \frac{|\delta b|}{|b|} $$

with equality when $b$ lies along $u_1$ and $\delta b$ along $u_2$: the worst case. With the two swapped, the answer is off by only $1/\kappa$ times as much.

The determinant does not measure this: $|\det A| = \sigma_1 \sigma_2$, how $A$ scales areas, is a product, while $\kappa$ is a ratio. With $I$ the matrix that changes nothing, $A = 0.01 I$ shrinks every arrow to a hundredth: its determinant is 0.0001, but $\kappa = 1$.

Ridge solves $(A^\top A + \lambda I)x = A^\top b$ instead, with $\lambda > 0$ ($A^\top$ is $A$ with rows and columns swapped). It shrinks the part of the answer along each $v_k$ by the factor $\sigma_k^2 / (\sigma_k^2 + \lambda)$: hardly at all where $\sigma_k$ is large, much where it is small, which is where errors are blown up. Its answer is steadier, but a little wrong even with no error.

## An example

Take

$$ A = \begin{bmatrix} 1 & 0.8 \\ 0.8 & 1 \end{bmatrix} $$

It sends $(1, 1)$ to $(1.8, 1.8)$ and $(1, -1)$ to $(0.2, -0.2)$, so it stretches these two directions at right angles by $\sigma_1 = 1.8$ and $\sigma_2 = 0.2$, and $\kappa = 9$. Let $x = (1, 1)$, so $b = (1.8, 1.8)$, and let the reading be off by $\delta b = (0.1, -0.1)$. Then $\delta x = (0.5, -0.5)$, since $A$ sends that to $(0.1, -0.1)$. Every length here carries a $\sqrt{2}$, which cancels: $b$ is off by $0.1 / 1.8 \approx 0.056$ of its size, and $x$ by $0.5 / 1 = 0.5$ of its size. That is 9 times as much: the worst case.

# In the picture

- White: $b$ and $x = A^{-1} b$.
- Pink: the error $\delta b$ in the reading, and the error $\delta x$ it makes.
- Yellow: the circle of every reading within $\varepsilon$ of $b$, and the ellipse of their answers.
- Blue: the two equations as lines, each with the band it can be in.
- Orange: $u_1$ and $v_1$; purple: $u_2$ and $v_2$. Green, when $\lambda > 0$: ridge.

> Try first "the worst case": $b$ along $u_1$ and the error along $u_2$, and $x$ is off by exactly $\kappa$ times as much as $b$.

see: elim, det, svd, overdet
