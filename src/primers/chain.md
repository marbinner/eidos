# The idea

Much of mathematics, and nearly every neural network, is built from maps done one after another. This scene follows a small change through such a chain. Each map stretches or shrinks a small change by its slope, so along the chain the slopes multiply: that is the chain rule. With two variables the slope of a map is a matrix, its Jacobian, and the Jacobian of the whole chain is the product of the Jacobians. Read backwards, the same product carries a gradient from the end of the chain to its start, which is what backpropagation does. You need derivatives and a matrix times a vector.

# The mathematics

## One variable

Do $f$ and then $g$: $z = g(f(x))$. Near a point $p$, a small step $\mathrm{d}x$ becomes a step of about $f'(p)\,\mathrm{d}x$ at $f(p)$, and that becomes about $g'(f(p))\,f'(p)\,\mathrm{d}x$ at $g(f(p))$. So

$$ (g \circ f)'(p) = g'\bigl(f(p)\bigr)\,f'(p) $$

The slope of $g$ is taken at $f(p)$, where the point has got to, not at $p$.

Example: $f(x) = x^3$ and $g(y) = \arctan y$, whose slope is $1/(1 + y^2)$. At $p = 1$: $f'(1) = 3$ and $f(1) = 1$, then $g'(1) = 1/2$, so the chain's slope is $3 \times 1/2 = 1.5$. Check it directly: the slope of $\arctan(x^3)$ is $3x^2/(1 + x^6)$, which is $3/2$ at $x = 1$.

## Two variables: Jacobians multiply

A map of the plane, $f(x) = (f_1(x), f_2(x))$, has at $p$ the Jacobian $J_f(p)$: the $2 \times 2$ matrix whose row $i$, column $j$ holds $\partial f_i / \partial x_j$. Near $p$, $f(p + \mathrm{d}x) \approx f(p) + J_f(p)\,\mathrm{d}x$. Do it twice:

$$ g\bigl(f(p + \mathrm{d}x)\bigr) \approx g\bigl(f(p)\bigr) + J_g\bigl(f(p)\bigr)\,J_f(p)\,\mathrm{d}x $$

so the Jacobian of the chain is $J = J_g(f(p))\,J_f(p)$, with $J_g$ on the left because it acts second. Pushing one step forwards gives $J\,\mathrm{d}x$; with $\mathrm{d}x = (1, 0)$ it is the first column of $J$.

## Backwards: transposes

Let a final number depend on the end of the chain, $L = u \cdot z$; its gradient there is $u$. A step $\mathrm{d}x$ changes $L$ by about $u \cdot (J\,\mathrm{d}x)$, and for any matrix

$$ u \cdot (J\,\mathrm{d}x) = (J^\top u) \cdot \mathrm{d}x $$

since both are $\sum_{i,j} u_i J_{ij}\,\mathrm{d}x_j$. So the gradient of $L$ as it depends on $x$ is $J^\top u = J_f^\top (J_g^\top u)$: pull $u$ back through $J_g^\top$, then through $J_f^\top$. Wherever a gradient is not zero, it stands at right angles to the level line of $L$ through its point. The two ends give the same number for the effect of $\mathrm{d}x$, but only to first order: for a smooth chain the true change of $L$ differs by an amount that shrinks at least as fast as the square of the step.

## Why networks are differentiated backwards

For a function of $n$ inputs and one output, forwards finds one slope per sweep, one input direction at a time: $n$ sweeps for the whole gradient. Backwards finds all $n$ in one sweep. A network may have millions of weights, and it has one loss. And a product is no larger than its factors allow, $\|J_g J_f\| \le \|J_g\|\,\|J_f\|$, where $\|J\|$ is the most $J$ lengthens any step: one flat link, such as a layer of tanh units whose inputs are all far from 0 and whose weights are not large, flattens the whole chain unless the other links stretch a great deal.

# In the picture

- Three planes, $x$, $y = f(x)$ and $z = g(y)$; the grey grid of the first is carried along by $f$ and then $g$. The white dot is $p$.
- Cyan: the step $\mathrm{d}x$ and its pushes $J_f\,\mathrm{d}x$ and $J_g J_f\,\mathrm{d}x$. Yellow: where the step really goes.
- Pink: $u$ and its pulls $J_g^\top u$ and $J_f^\top J_g^\top u$. Orange: level lines of $L$, bright through each point.
- The board writes out $J_g$, $J_f$ and their product, and both sweeps with their numbers. With one variable, three number lines replace the planes.

> Try first "slopes multiply", then "the same number from either end".

see: calc1, jacobian, compose, autodiff
