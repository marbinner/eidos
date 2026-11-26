# The idea

The gradient says which way a surface $z = f(x, y)$ slopes; the second derivatives say how that slope changes, which is how the surface curves. Collected in a matrix, the Hessian, they tell bottoms, tops and mountain passes apart, and Newton's method uses them to jump towards such level spots. You need the gradient and the eigenvalues of a symmetric matrix (the scenes "surface" and "quadform").

# The mathematics

## The Hessian

Differentiate twice: $f_{xx}$ is how fast $\partial f/\partial x$ changes with $x$, $f_{yy}$ likewise for $y$, and $f_{xy}$ how fast $\partial f/\partial x$ changes with $y$. The Hessian is

$$ H = \begin{bmatrix} f_{xx} & f_{xy} \\ f_{xy} & f_{yy} \end{bmatrix} $$

It is symmetric because $f_{xy} = f_{yx}$ wherever the second derivatives are continuous. The second derivative along a unit direction $u$, what the scene calls the curvature that way, is $u^\top H u$.

## The paraboloid that fits

Taylor's formula to second order, for a step $d$ from $p$:

$$ f(p + d) \approx f(p) + \nabla f \cdot d + \tfrac{1}{2}\, d^\top H d $$

The right side has the value, slopes and second derivatives of $f$ at $p$. Its graph is a paraboloid (a trough, ridge or plane where $H$ is singular); with continuous second derivatives, its error is small even against $|d|^2$ as $d$ shrinks.

As $H$ is symmetric, its eigenvalues $\lambda_1 \ge \lambda_2$ are real and it has eigenvectors at right angles. Along a unit eigenvector $v$ the curved term is $\tfrac{1}{2} \lambda t^2$ at $d = tv$, and every other direction's curvature lies between $\lambda_2$ and $\lambda_1$: the eigenvectors point the ways of most and least curvature.

## The second-derivative test

Where $\nabla f = 0$ (a stationary point) the plane term is gone, and unless an eigenvalue is 0 the curved term decides what $p$ is:

- both eigenvalues positive: a bowl, and $p$ is a minimum;
- both negative: a dome, a maximum;
- one of each sign: a saddle point;
- one of them 0: the test says nothing. $x^2 + y^4$ and $x^2 - y^4$ have the same $H$ at the origin, where the first has a minimum and the second a saddle.

## Newton's method

The paraboloid is flat where its own gradient, $\nabla f + H d$, is 0. Newton's method steps there, and repeats:

$$ d = -H^{-1} \nabla f $$

This needs $H$ to have no eigenvalue 0. On a quadratic $f$, one step lands on the stationary point. Started close enough to one where $H$ has no eigenvalue 0 (for smooth enough $f$), each distance from it is at most a fixed multiple of the square of the last: the correct digits at least roughly double. But it seeks any point where $\nabla f = 0$, a saddle or top as readily as a bottom; far from them, or where $H$ is nearly singular, a step can go anywhere.

Gradient descent, $x \leftarrow x - h \nabla f(x)$, needs no $H$. On a quadratic bowl each step multiplies the distance along an eigenvector by $1 - h\lambda$: it settles only if $h < 2/\lambda_1$, and then crawls if $\lambda_2$ is much smaller.

Example, the two wells $f = (x^2 - 1)^2 + y^2$ at $p = (0.35, 0.9)$: $\nabla f = (4x^3 - 4x, 2y) = (-1.2285, 1.8)$, $f_{xx} = 12x^2 - 4 = -2.53$, $f_{yy} = 2$, $f_{xy} = 0$, so the paraboloid is a saddle. Newton's step $(-0.49, -0.9)$ goes to $(-0.14, 0)$; two more nearly reach the pass, $(0, 0)$.

# In the picture

- The surface is $z = f(x, y)$; the map shows it from above.
- The white point is $p$; the white sheet is the fitted paraboloid, dotted on the map its level curves.
- The card gives $\nabla f$, $H$, its eigenvalues and what they say.
- Orange and purple: the eigenvectors of $\lambda_1$ and $\lambda_2$, as a cross on the map, and the paraboloid (thick) and surface (thin) along each.
- Green: Newton's step and its run; cyan: gradient descent's run.

> Try first "a bowl, a dome and a saddle": three stationary points, told apart by the signs of $\lambda_1$ and $\lambda_2$.

see: surface, quadform, descent
