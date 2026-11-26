# The idea

A path runs across a hillside: where on the path is it highest? The hill is a function $f(x, y)$, its height at the point $(x, y)$, and the path is a curve $g(x, y) = c$. Its highest point is seldom the hill's top, so setting the slope of $f$ to zero will not find it. Lagrange's method turns the question into equations, whose multiplier $\lambda$ says what the path costs. It needs gradients and contour lines.

# The mathematics

## Why the two gradients line up

The gradient $\nabla f = (\partial f/\partial x, \partial f/\partial y)$ points straight uphill, at right angles to the contour line of $f$, and a small step $d$ changes $f$ by about $\nabla f \cdot d$. The path is a contour line of $g$, so $\nabla g$ is at right angles to it. With $T$ the unit arrow along the path, split $\nabla f$ into a part across the path and a part along it:

$$ \nabla f = \lambda \nabla g + h\,T, \qquad \lambda = \frac{\nabla f \cdot \nabla g}{|\nabla g|^2}, \quad h = \nabla f \cdot T $$

If $h$ is not 0, one way along the path is uphill, and the point is not the highest. So at the highest point $h = 0$: the path touches the contour line of $f$ there instead of crossing it, and

$$ \nabla f = \lambda \nabla g, \qquad g(x, y) = c $$

Three equations for $x$, $y$ and $\lambda$. Where $\nabla g = 0$ the path may have a corner, and the method says nothing.

## Candidates, not answers

The lowest point, and any point where $f$ stops changing along the path, satisfies them too: they give candidates, and comparing $f$ at them says which is highest.

## The multiplier is a price

Write $f^*(c)$ for the highest value of $f$ on the path $g = c$. Raise $c$ by a little, $\delta c$: the new path lies a distance $\delta c / |\nabla g|$ further out, along $\nabla g$. Stepping there from the old highest point raises $f$ by about $\nabla f \cdot \nabla g \, \delta c / |\nabla g|^2 = \lambda \, \delta c$; moving along the new path to its own highest point changes $f$ only by an amount of the size of $\delta c^2$, because $f$ had stopped changing along the path. So, wherever the highest point moves smoothly with $c$,

$$ \frac{\mathrm{d} f^*}{\mathrm{d} c} = \lambda $$

Its sign depends on how the constraint is written: written $-g = -c$, the same path gives $-\lambda$.

Example: $f = x + 2y$ on the circle $x^2 + y^2 = c$. Here $\nabla f = (1, 2)$ and $\nabla g = (2x, 2y)$, so $\nabla f = \lambda \nabla g$ gives $x = 1/(2\lambda)$ and $y = 1/\lambda$. For $c = 1$, $x^2 + y^2 = 5/(4\lambda^2) = 1$, so $\lambda = \pm\sqrt{5}/2$. The plus sign gives the top, $(1, 2)/\sqrt{5}$, where $f = \sqrt{5} = 2.236$; the minus sign gives the bottom. For any $c$ the same steps give $f^*(c) = \sqrt{5c}$, whose slope at $c = 1$ is $\sqrt{5}/2 = 1.118$: the multiplier.

## Inequalities

If every point with $g(x, y) \le c$ is allowed, the best point is either inside, at the top of a hill, where $\nabla f = 0$ and $\lambda = 0$, or on the edge, with $\nabla f = \lambda \nabla g$ and $\lambda \ge 0$. A negative $\lambda$ at the highest point of the edge says that inwards is uphill.

# In the picture

- The map is $f$, lighter where higher, with its contour lines.
- The cyan curve is the path, and the cyan arrow $\nabla g$.
- The white line is the contour line of $f$ through the point.
- Yellow is $\nabla f$, pink its part along the path, $h\,T$, and the dashed line its part across, $\lambda \nabla g$.
- Green dots are tops, orange ones bottoms; the charts show $f$ along the path, and $f^*(c)$ with a purple line of slope $\lambda$.

> Try first "slide to the top": the pink arrow shrinks to nothing as the yellow one swings into line with the cyan.

see: surface, underdet, descent
