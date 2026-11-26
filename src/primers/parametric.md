# The idea

A curve or a surface can be made by a rule: one number in and a point out makes a curve, two numbers in and a point in space out make a surface. As the numbers run over their ranges, the points sweep out the shape. Made this way, a shape can do what the graph of a function never does: close up into a tube, tie itself in a knot, or join up as a band with only one side. You need sines and cosines and the idea of a function.

# The mathematics

## A curve from one number

A rule that sends a number $u$ to a point gives a curve: as $u$ runs over its range, the point moves along it. For example,

$$ u \mapsto (\cos u,\ \sin u), \qquad 0 \le u \le 2\pi $$

goes once round the circle of radius $1$, starting from $(1, 0)$. The number $u$ is called the parameter. The graph of a function $y = f(x)$ is the special case $u \mapsto (u, f(u))$, but a curve made by a rule may also turn back, cross itself or close up.

## A surface from two numbers

With two numbers $(u, v)$ in and a point $(x, y, z)$ out, the rule makes a surface. Hold $v$ fixed and let $u$ run: the point traces a curve on the surface. Hold $u$ fixed and let $v$ run: another curve. These are the parameter lines, and together they cover the surface like a net. The torus, the scene's first shape, is

$$ x = (2 + a\cos v)\cos u, \quad y = (2 + a\cos v)\sin u, \quad z = a\sin v $$

with $u$ and $v$ each running from $0$ to $2\pi$. Hold $u$ fixed: the point is $2 + a\cos v$ out from the vertical axis and $a\sin v$ up, so it goes round a circle of radius $a$ whose centre is $2$ out, a cross-section of the tube. Hold $v$ fixed: the point goes round the vertical axis, a circle round the hole.

Example: with $a = 1$, $(u, v) = (0, 0)$ gives $(3, 0, 0)$, the outermost point; $(0, \pi)$ gives $(1, 0, 0)$, the innermost; and $(0, \pi/2)$ gives $(2, 0, 1)$, the top of the tube. With $a = 2$, $v = \pi$ gives $(0, 0, 0)$ for every $u$: the inside of the tube reaches the axis, and the hole has closed to a single point.

## Twists, and graphs again

The Möbius band is

$$ x = (2 + v\cos\tfrac{u}{2})\cos u, \quad y = (2 + v\cos\tfrac{u}{2})\sin u, \quad z = v\sin\tfrac{u}{2} $$

with $u$ from $0$ to $2\pi$ along the band and $v$ from $-0.8$ to $0.8$ across it. Going once round, the half angle $u/2$ turns the line across the band through half a turn. At the end, $u = 2\pi$, the point for $v$ is $(2 - v, 0, 0)$, which is where the start, $u = 0$, put $-v$. The band meets itself turned over, and so has only one side.

The wave sheet has $x = u$ and $y = v$, and $z$ a formula in $u$ and $v$: it is the graph of a function of two numbers, $z = f(x, y)$, made as a parametric surface. Where a formula has no value, such as the square root of a negative number, the surface has a hole.

# In the picture

- The surface is every point the rule gives. Its colour is the height $z$ at first, or $u$, $v$ or the distance from the centre, as you choose.
- The dark lines are the parameter lines, 36 each way: along each one, $u$ or $v$ is held fixed.
- The boxes hold the formulas for $x$, $y$ and $z$, with the ranges of $u$ and $v$ below them; the slider is $a$, and $t$ is the scene's clock.
- Each shape is scaled to fit the view, so the picture shows its form but not its size.

> Try first "a band with one side": follow one colour once round the band, and see it come back as its opposite.

see: graph, surface, fourier
