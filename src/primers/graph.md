# The idea

A function takes a number and gives back another: $f(x) = x^2$ turns $3$ into $9$. Its graph draws all these pairs at once, each as the point $(x, f(x))$, so you can see at a glance where it is large or small, where it rises or falls, and where it turns. This scene draws the graph of any formula you type, several at once, and lets numbers in the formulas move. You need to know what a function is, and a derivative as a rate of change.

# The mathematics

## A curve of points

For each $x$ in view the scene works out $y = f(x)$ and marks the point $(x, y)$: along the horizontal axis to $x$, then up by $y$, or down if $y$ is negative. Joined up, these points are the curve. Reading a graph is the same in reverse: go along to $x$, then up or down to the curve; that height is $f(x)$. Where the curve crosses the horizontal axis, $f(x) = 0$.

Example: the second curve is the bell

$$ y = \exp\left(-\frac{x^2}{2b^2}\right) $$

where $\exp(s)$ means $e^s$. At $x = 0$ it is $e^0 = 1$, its top. At $x = b$ and at $x = -b$ it is $e^{-1/2} \approx 0.61$, whatever $b$ is: $b$ measures the bell's width.

A function of two numbers, $z = f(x, y)$, has a graph too: a surface over the $(x, y)$ plane, with height $f(x, y)$.

## Slope

How steeply the curve rises at a point is its slope there, the derivative $f'(x)$: the rate at which $y$ changes as $x$ grows. The tangent is the straight line through the point with that slope. Where a smooth curve turns from rising to falling, or back, its tangent lies flat: $f'(x) = 0$.

Example: the third curve,

$$ y = 0.1x^3 - bx, \qquad f'(x) = 0.3x^2 - b $$

has a flat tangent where $x^2 = b / 0.3 = 10b/3$. At $b = 1$ that is $x = \pm\sqrt{10/3} \approx \pm 1.83$, its two turns. For $b < 0$ the slope is positive everywhere, so the curve rises all along and has no turns.

## Sliders and time

A formula with $a$, $b$ or the clock $t$ in it is a whole family of curves, one for each value. Two rules, for any function $g$, make most changes easy to read:

- $g(x - h)$ is the curve of $g$ moved right by $h$, since at $x + h$ it takes the value $g$ had at $x$.
- $g(ax)$, for $a > 1$, is the curve of $g$ squeezed towards the vertical axis by the factor $a$, since at $x / a$ it takes the value $g$ had at $x$.

Example: $\sin(ax - t) = \sin\big(a(x - t/a)\big)$ is the wave $\sin(ax)$ moved right by $t/a$. Its crests are $2\pi/a$ apart and move right by $1/a$ for each unit that $t$ grows: half a unit at $a = 2$. At $a = 1$ the waves are twice as long and travel twice as fast. The first curve divides this wave by $1 + 0.1x^2$, which shrinks its height away from the middle: to $1/1.9 \approx 0.53$ of it at $x = 3$.

# In the picture

- Each coloured curve is the graph of one formula, named by that formula at the right-hand edge: cyan for curve 1, pink for curve 2, yellow for curve 3.
- The grey vertical line follows the pointer. Where it meets a curve, the number beside the dot gives $f(x)$ there; with "show the tangent under the pointer" it also gives the slope $f'(x)$, and the tangent is drawn.
- The sliders set $a$ and $b$. The clock $t$ runs while the scene does.

> Try first "where a cubic turns": move the pointer along the curve and watch the slope pass through 0 at each turn.

see: calc1, surface, parametric
