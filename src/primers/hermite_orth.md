# The idea

Two arrows are perpendicular when their dot product is zero. Functions can be perpendicular too, once we choose what plays the part of the dot product; here it is the average of their product when the input is a standard normal number. The scene draws that average as an area, runs Gram–Schmidt on the powers $1, x, x^2, \ldots$ to build the Hermite polynomials, and shows by integration by parts why different degrees cancel exactly. You need Gram–Schmidt, the normal distribution and integration by parts.

# The mathematics

## A dot product for functions

With $\varphi(x) = e^{-x^2/2} / \sqrt{2\pi}$ the density of a standard normal $X$,

$$ \langle f, g \rangle = \mathbb{E}[f(X)\, g(X)] = \int f g \, \varphi \, \mathrm{d}x $$

It behaves like the dot product of arrows: it is symmetric, it can be split over sums, and $\langle f, f \rangle \ge 0$. Its square root $\sqrt{\langle f, f \rangle}$, the root mean square, plays the part of length. The function $f g \varphi$ is a curve, and $\langle f, g \rangle$ is its signed area: the area above the axis minus the area below. The functions are orthogonal when that is zero.

## The powers overlap

Odd powers of $X$ average to 0 and even ones to $\mathbb{E}[X^{2k}] = 1 \cdot 3 \cdot 5 \cdots (2k - 1)$, so $\mathbb{E}[X^4] = 3$, $\mathbb{E}[X^6] = 15$, $\mathbb{E}[X^8] = 105$. Divided by their root mean squares, $x^2$ and $x^4$ become $x^2/\sqrt{3}$ and $x^4/\sqrt{105}$, and

$$ \left\langle \frac{x^2}{\sqrt{3}}, \frac{x^4}{\sqrt{105}} \right\rangle = \frac{15}{\sqrt{3}\, \sqrt{105}} = 0.845 $$

Both are never negative, so nothing cancels. Powers of opposite parity do give 0, by symmetry, but among the first ten the overlaps of the same parity reach 0.94.

## Gram–Schmidt

Take the powers in turn. From $x^n$ subtract its component $\langle x^n, h_k \rangle h_k$ along each function $h_k$ already made, and divide what is left by its root mean square. The results are the normalised Hermite polynomials, $h_0 = 1$, $h_1 = x$, $h_2 = (x^2 - 1)/\sqrt{2}$, and in general $h_n = \mathrm{He}_n / \sqrt{n!}$, with $\mathrm{He}_n$ the monic ones of the family scene. Their table of overlaps is 1 on the diagonal and 0 everywhere else.

## Why it is exactly zero

The tool is integration by parts under the Gaussian,

$$ \mathbb{E}[f'(X)\, g(X)] = \mathbb{E}\left[f(X) \left(X g(X) - g'(X)\right)\right] $$

together with two rules from the family scene: lowering, $h_n' = \sqrt{n}\, h_{n-1}$, and raising, $X h_{m-1} - h_{m-1}' = \sqrt{m}\, h_m$. Write $h_m$ by raising and move the derivative across:

$$ \langle h_m, h_n \rangle = \frac{1}{\sqrt{m}}\, \mathbb{E}\left[(X h_{m-1} - h_{m-1}')\, h_n\right] = \frac{1}{\sqrt{m}}\, \mathbb{E}[h_{m-1}\, h_n'] = \sqrt{\frac{n}{m}}\, \langle h_{m-1}, h_{n-1} \rangle $$

Each step lowers both degrees by one and multiplies by a known factor. If $m < n$, after $m$ steps the first function is $h_0 = 1$ and what is left is $\mathbb{E}[h_{n-m}(X)]$, the average of a Hermite polynomial of degree at least 1. That average is 0: the rule with $f = 1$ gives $\mathbb{E}[X g - g'] = 0$ for every $g$, and every $h_k$ with $k \ge 1$ is of that form.

Example: $\langle h_3, h_5 \rangle = \sqrt{5/3} \cdot \sqrt{4/2} \cdot \sqrt{3/1}\, \langle h_0, h_2 \rangle = \sqrt{10}\, \langle h_0, h_2 \rangle$, and $\langle h_0, h_2 \rangle = (\mathbb{E}[X^2] - 1)/\sqrt{2} = 0$. With equal degrees every factor is 1, and the ladder ends at $\langle 1, 1 \rangle = 1$.

# In the picture

- Pink is the first function and green the second, each times $\sqrt{\varphi}$.
- Below them is their product $f g \varphi$: cyan where it is above the axis, orange where it is below.
- The yellow curve is the area so far, from left to right, counting orange as negative; it ends at $\langle f, g \rangle$.
- The table holds $\langle f, g \rangle$ for every pair up to degree 9: red is 1, paler is smaller, dark is exactly 0. The white box is the pair plotted; click a cell to pick another.
- The slider "steps down the ladder" moves the pair down the diagonal (yellow arrow), with the factor read out in the panel.

> Try first "plain powers", then "after Gram–Schmidt": the same pair of degrees, before and after.

see: gram, hermite, hermite_expand
