# The idea

A map of the plane takes each point $(x, y)$ to a point $(u, v) = f(x, y)$, and may bend straight lines into curves. Yet near a point $p$ where it is smooth, what it does to a small step from $p$ is nearly a linear map: a $2 \times 2$ matrix, the Jacobian $J$, made of the four partial derivatives at $p$. It is the derivative of the map, as a slope is the derivative of a curve. You need to know what a partial derivative is (scene surface) and what a matrix does to the plane (scene linmap2).

# The mathematics

## The best linear map at a point

The map is two functions, $u(x, y)$ and $v(x, y)$. For a small step $h = (h_1, h_2)$ from $p$, each changes by about its slope along $x$ times $h_1$ plus its slope along $y$ times $h_2$. For both at once:

$$ f(p + h) \approx f(p) + Jh, \qquad J = \begin{bmatrix} \partial u/\partial x & \partial u/\partial y \\ \partial v/\partial x & \partial v/\partial y \end{bmatrix} $$

with the partial derivatives taken at $p$. Multiply $J$ by the step $(1, 0)$ and you get its first column: where a small step along $x$ goes. The second column is where a step along $y$ goes. So $J$ takes a small square of side $s$ round $p$ to a parallelogram with sides $s$ times its columns.

## How close it is

If $f$ has continuous second derivatives, what $Jh$ leaves out is, near $p$, at most a fixed number times $|h|^2$. So for such a square, the largest misfit between its true image and the parallelogram, as a share of $s$, is at most a fixed number times $s$. Halve a small patch and the share nearly halves, if the second derivatives at $p$ are not all $0$.

Example: the bend $f(x, y) = (x + 0.4y^2,\ y + 0.4x^2)$ has

$$ J = \begin{bmatrix} 1 & 0.8y \\ 0.8x & 1 \end{bmatrix} $$

so at $p = (0.6, -0.45)$ its columns are $(1, 0.48)$ and $(-0.36, 1)$. What is left over is exactly $(0.4h_2^2,\ 0.4h_1^2)$. At a corner of a patch of side $s = 0.7$, where $h_1$ and $h_2$ are $\pm 0.35$, its length is $0.4 \times 0.35^2 \times \sqrt{2} = 0.069$, or 9.9 % of the side. With $s = 0.35$ it is 4.9 %.

## The determinant

A linear map multiplies every area by the same number, $|\det J|$, where $\det J = ad - bc$ for $J$ with rows $(a, b)$ and $(c, d)$ (scene det). So a small patch near $p$, of area $A$, lands on a patch of area about $|\det J|\,A$.

- If $\det J < 0$, the patch is also turned over: an arrow going anticlockwise round it comes out going clockwise.
- If $\det J = 0$, $J$ presses the plane onto a line or a point. Where $\det J \ne 0$ (and the partial derivatives are continuous), $f$ can be undone near $p$; where $\det J = 0$, no map with a Jacobian can undo it near $p$.

Polar coordinates, $(r, \theta) \mapsto (r\cos\theta,\ r\sin\theta)$, have $\det J = r$: a cell twice as far out is twice as large. Squaring a complex number, $(x, y) \mapsto (x^2 - y^2,\ 2xy)$, has $J$ with rows $(2x, -2y)$ and $(2y, 2x)$: a turn and a stretch, so small squares stay nearly square (except at $0$, where $J = 0$). The fold $(x, y) \mapsto (x, y^2)$ has $\det J = 2y$: zero on the crease $y = 0$, negative below it.

# In the picture

- White, left: the patch, of side $s$, round $p$ (drag the dot). The blue grid is carried to the right by $f$; both pictures share one scale.
- Yellow: where $f$ takes the patch. Dashed: where $J$ takes it, the points $f(p) + Jh$.
- Pink and green: steps of $s/2$ along $x$ and $y$, and where $J$ takes them: $s/2$ times $J$'s columns.
- Red: the misfit, where the two outlines are furthest apart.

> Try first "zoom in: curved becomes straight", and watch the yellow outline settle onto the dashed one.

see: surface, linmap2, det, chain
