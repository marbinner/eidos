# The idea

A symmetric matrix $A$, one with the same entry above and below the diagonal, gives every point $x$ of the plane a number, its height $x^\top A x$. Over the plane the heights make a bowl, a saddle or a trough (or one upside down), with the eigenvectors of $A$ as axes. Near a lowest point a smooth function of two numbers usually looks like such a bowl, and its shape sets how fast gradient descent reaches the bottom. You need eigenvalues and eigenvectors, and the transpose $x^\top$: the column $x$ written as a row.

# The mathematics

## The height

$$ A = \begin{bmatrix} a & b \\ b & c \end{bmatrix}, \qquad x^\top A x = a x_1^2 + 2b x_1 x_2 + c x_2^2 $$

Here $x = (x_1, x_2)$, and $x^\top A x$ is the dot product $x \cdot Ax$. Along a unit eigenvector $v$, with $Av = \lambda v$, the point $tv$ has height $\lambda t^2$: a parabola, opening upwards if $\lambda > 0$ and downwards if $\lambda < 0$.

## Turning the axes onto the eigenvectors

A symmetric matrix has real eigenvalues $\lambda_1 \ge \lambda_2$, and unit eigenvectors $v_1$, $v_2$ at right angles. Write $x = y_1 v_1 + y_2 v_2$: $y_1$ and $y_2$ are the coordinates along the turned axes. Then $Ax = \lambda_1 y_1 v_1 + \lambda_2 y_2 v_2$, and in its dot product with $x$ the mixed terms vanish, because $v_1 \cdot v_2 = 0$:

$$ x^\top A x = \lambda_1 y_1^2 + \lambda_2 y_2^2 $$

In matrices this is the spectral theorem: $A = Q \Lambda Q^\top$, with $v_1$ and $v_2$ the columns of $Q$, the eigenvalues on the diagonal of $\Lambda$, and $y = Q^\top x$. So the signs of the eigenvalues give the shape:

- both above 0: a bowl, every way out of $0$ goes up (positive definite); both below 0, the bowl upside down (negative definite);
- one of each sign: a saddle (indefinite);
- one of them 0: a trough along a whole line (semidefinite).

On the unit circle $y_1^2 + y_2^2 = 1$, since turning keeps lengths, so the height there is an average of $\lambda_1$ and $\lambda_2$ with weights $y_1^2$ and $y_2^2$: never above $\lambda_1$, never below $\lambda_2$, and equal to them on the axes. For a bowl, the level curve $x^\top A x = 1$ is an ellipse with half-axes $1/\sqrt{\lambda_1}$ and $1/\sqrt{\lambda_2}$.

Example, the scene's first matrix: $a = 1.7$, $b = 0.6$, $c = 0.8$. It sends $(2, 1)$ to $(4, 2)$ and $(-1, 2)$ to $(-0.5, 1)$, so $\lambda_1 = 2$ along $(2, 1)$, at 26.6° to the $x_1$ axis, and $\lambda_2 = 0.5$ at right angles. At $x = (1, 0)$ the height is $a = 1.7$. In turned coordinates it is $y_1 = 2/\sqrt{5}$, $y_2 = -1/\sqrt{5}$, and $2 \cdot 4/5 + 0.5 \cdot 1/5 = 1.7$ again, between $0.5$ and $2$.

## Going downhill

The gradient, the arrow straight uphill, is $\nabla(x^\top A x) = 2Ax$, and a step of gradient descent, $x \leftarrow x - h \cdot 2Ax$, multiplies $y_1$ by $1 - 2h\lambda_1$ and $y_2$ by $1 - 2h\lambda_2$. In a bowl the step size $h$ must stay below $1/\lambda_1$, or $y_1$ never shrinks; then, if $\lambda_2$ is much smaller, $y_2$ shrinks slowly. In "a narrow valley" ($\lambda_1 = 3$, $\lambda_2 = 0.1$, $h = 0.3$) it is multiplied by $0.94$ a step.

# In the picture

- The surface: the height $x^\top A x$ above each point $x$. The floor: the same heights as colours, with the same level curves.
- Orange: $v_1$'s axis on the floor, and above it the parabola $\lambda_1 t^2$. Purple: the same for $\lambda_2$.
- White ring: the unit circle lifted to its heights, with $x$ the dot on it, and its height graphed against $x$'s angle.
- Yellow: the level curve $x^\top A x = 1$ (and $-1$ for a saddle).
- Pale arrows: the axes $x_1$ and $x_2$ (once turned, $y_1$ and $y_2$). Cyan: gradient descent.

> Try first "round the circle": the ring rises to $\lambda_1 = 2$ on the orange axis and falls to $\lambda_2 = 0.5$ on the purple one, twice a turn.

see: eigen, surface, descent, pca
