# The idea

The powers $1, x, x^2, \ldots$ are the plainest building blocks for functions, but they overlap. The Hermite polynomials are what is left when each power is made perpendicular to all the earlier ones, where two functions are perpendicular when the average of their product is zero for a standard normal number $X$. They make averages over the bell curve easy to work out, degree by degree. You need derivatives, integration by parts, the normal distribution and Gram–Schmidt.

# The mathematics

## Averages under the bell

The Gaussian weight is the density of a standard normal number:

$$ \varphi(x) = \frac{e^{-x^2/2}}{\sqrt{2\pi}}, \qquad \mathbb{E}[g(X)] = \int g(x)\, \varphi(x)\, \mathrm{d}x $$

Odd powers average to 0, as the bell is symmetric; $\mathbb{E}[X^2] = 1$ and $\mathbb{E}[X^4] = 3$.

## Gram–Schmidt on the powers

Take the powers in turn and subtract from each its components along the ones already done, keeping the leading coefficient 1 (monic). The results are the probabilists' Hermite polynomials $\mathrm{He}_n$. $\mathrm{He}_0 = 1$ and $\mathrm{He}_1 = x$, already perpendicular to $1$ since $\mathbb{E}[X] = 0$. For $x^2$ the component along $1$ is $\mathbb{E}[X^2] = 1$ and the one along $x$ is $\mathbb{E}[X^3]/\mathbb{E}[X^2] = 0$, so $\mathrm{He}_2 = x^2 - 1$. Continuing:

$$ \mathrm{He}_3 = x^3 - 3x, \quad \mathrm{He}_4 = x^4 - 6x^2 + 3 $$

Their mean squares grow: $\mathbb{E}[\mathrm{He}_2(X)^2] = 3 - 2 + 1 = 2$, and in general $\mathbb{E}[\mathrm{He}_n(X)^2] = n!$. So the scene's default scaling, $h_n = \mathrm{He}_n / \sqrt{n!}$, has mean square 1 at every degree.

## One rule behind every identity

Integrate $\int f' g \varphi \, \mathrm{d}x$ by parts, using $\varphi' = -x \varphi$; for polynomials $f$ and $g$ the ends vanish, as $\varphi$ dies away faster than they grow:

$$ \mathbb{E}[f'(X)\, g(X)] = \mathbb{E}\left[f(X) \left(X g(X) - g'(X)\right)\right] $$

With it, and the fact that a polynomial perpendicular to all lower degrees is a multiple of the Hermite polynomial of its degree:

- Raising: $x\,\mathrm{He}_n - \mathrm{He}_n' = \mathrm{He}_{n+1}$. The left side is monic of degree $n + 1$, and for $p$ of lower degree the rule gives $\mathbb{E}[(X\,\mathrm{He}_n - \mathrm{He}_n')\, p] = \mathbb{E}[\mathrm{He}_n\, p'] = 0$, as $p'$ has degree below $n$.
- Lowering: $\mathrm{He}_n' = n\,\mathrm{He}_{n-1}$. The slope $\mathrm{He}_n'$ has degree $n - 1$ and leading coefficient $n$, and for $p$ of degree below $n - 1$, $\mathbb{E}[\mathrm{He}_n' p] = \mathbb{E}[\mathrm{He}_n (X p - p')] = 0$. Normalised, $h_n' = \sqrt{n}\, h_{n-1}$.
- Lowering put into raising gives the three-term rule, $\mathrm{He}_{n+1} = x\,\mathrm{He}_n - n\,\mathrm{He}_{n-1}$; raising both sides of lowering gives the eigenfunction rule, $\mathrm{He}_n'' - x\,\mathrm{He}_n' = -n\,\mathrm{He}_n$.
- The $n!$: $\mathbb{E}[\mathrm{He}_n^2] = \mathbb{E}[\mathrm{He}_n (X\,\mathrm{He}_{n-1} - \mathrm{He}_{n-1}')] = \mathbb{E}[\mathrm{He}_n'\,\mathrm{He}_{n-1}] = n\, \mathbb{E}[\mathrm{He}_{n-1}^2]$.

Check at $n = 4$: $\mathrm{He}_4' = 4x^3 - 12x = 4\,\mathrm{He}_3$, and $x\,\mathrm{He}_4 - \mathrm{He}_4' = x^5 - 10x^3 + 15x = \mathrm{He}_5$.

## Zeros and the Gauss–Hermite rule

$\mathrm{He}_n$ has $n$ real zeros, and between two neighbouring zeros of $\mathrm{He}_n$ or of $\mathrm{He}_{n+1}$ lies exactly one zero of the other. Put a weight $w_i$ at each zero $x_i$ of $\mathrm{He}_n$, chosen so that the sum is right for every polynomial of degree below $n$. Then it is right up to degree $2n - 1$:

$$ \sum_{i=1}^{n} w_i\, p(x_i) = \mathbb{E}[p(X)] \quad \text{for } \deg p \le 2n - 1 $$

Why: divide, $p = q\,\mathrm{He}_n + r$ with $q$ and $r$ of degree below $n$. The part $q\,\mathrm{He}_n$ averages to 0 by orthogonality and is 0 at the nodes, so both sides see only $r$. With $n = 2$ the nodes are $\pm 1$, each of weight $1/2$: $x^2$ gives $1$, correct, but $x^4$ gives $1$ instead of $3$.

# In the picture

- The thick curve is degree $n$ in the chosen scaling, its formula at the top; thin curves are lower degrees.
- Purple is the bell $\varphi$, not to scale.
- Large dots on the axis are the zeros of degree $n$, small white dots those of degree $n + 1$.
- With the derivative shown, dashed white is the measured slope, wide yellow $\sqrt{n}\, h_{n-1}$ (or $n\,\mathrm{He}_{n-1}$).
- Yellow stems are the Gauss–Hermite nodes, each as tall as its weight divided by its share of the axis.

> Try first "the lowering rule": the slope of $h_4$ lies exactly on $2 h_3$.

see: gram, hermite_orth, hermite_expand
