# The idea

A function of two numbers, $z = f(x, y)$, can be drawn as a landscape: $(x, y)$ is a place on the ground and $z$ the height there. This scene asks of such a landscape what the derivative asks of a curve: how steep is it at a point, and which way is uphill? The answers are two slopes, the partial derivatives, and the arrow they make, the gradient. You need the derivative of a function of one number: the slope of its graph.

# The mathematics

## Slopes along the axes

Stand at the point $(x_0, y_0)$ and walk in the $x$ direction only, keeping $y$ fixed. The height then depends on $x$ alone, and the surface cut along that path is a curve. Its slope at the point is the partial derivative

$$ \frac{\partial f}{\partial x} = \lim_{h \to 0} \frac{f(x_0 + h, y_0) - f(x_0, y_0)}{h} $$

Walking in the $y$ direction, $x$ fixed, gives $\partial f/\partial y$ the same way. To work one out, treat the other letter as a fixed number and take the ordinary derivative.

## The tangent plane

Close up, a smooth surface looks flat: like the flat sheet that touches it at the point, the tangent plane. Its tilt along each axis is that axis's slope:

$$ z \approx f(x_0, y_0) + \frac{\partial f}{\partial x}\,(x - x_0) + \frac{\partial f}{\partial y}\,(y - y_0) $$

It holds the tangent lines of both cuts, and near the point it stays close to the surface.

## The gradient

The two slopes together make an arrow on the map, the gradient

$$ \nabla f = \left( \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \right) $$

Walk from the point in a direction $u = (u_1, u_2)$ of length 1. On the tangent plane the height changes at the rate

$$ u_1 \frac{\partial f}{\partial x} + u_2 \frac{\partial f}{\partial y} = |\nabla f| \cos\theta $$

where $\theta$ is the angle between $u$ and $\nabla f$ (the left side is their dot product, which is the two lengths times $\cos\theta$). This is largest when $\theta = 0$, with $u$ along $\nabla f$. So $\nabla f$ points straight uphill, and its length $|\nabla f|$ is the steepest slope. The rate is 0 when $u$ is at a right angle to $\nabla f$, and that is the way along the contour line, where the height does not change: the gradient crosses contour lines at right angles. Where $\nabla f = 0$ the tangent plane lies level, as at a top, a bottom, or a saddle (which curves up one way and down another).

Example: the bowl $f = 0.25(x^2 + y^2)$ at $(1.2, 0.9)$. With $y$ fixed, $f$ is $0.25x^2$ plus a fixed number, of slope $0.5x$ (the slope of $x^2$ is $2x$). So $\partial f/\partial x = 0.6$ and likewise $\partial f/\partial y = 0.45$, and

$$ |\nabla f| = \sqrt{0.6^2 + 0.45^2} = 0.75 $$

half the point's distance, $1.5$, from the centre. And $\nabla f = 0.5\,(x, y)$ points straight away from the bottom.

# In the picture

- The surface is $z = f(x, y)$, coloured by height, with contour lines; the map below is the same function seen from above.
- The white point is $(x_0, y_0)$: drag it on the map.
- The pink curve is the cut along $x$, of slope $\partial f/\partial x$; the green curve the cut along $y$, of slope $\partial f/\partial y$.
- The pale sheet is the tangent plane.
- The yellow arrow on the map is $\nabla f$, and on the surface the way uphill; the white arrows are the gradient everywhere, longer where steeper.

> Try first "a saddle point": the gradient is 0 and the tangent plane level, yet the pink cut curves up and the green one down.

see: calc1, descent, field
