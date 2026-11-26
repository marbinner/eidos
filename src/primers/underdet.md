# The idea

One equation in two unknowns, like $x + 2y = 2$, is solved by every point of a line. With fewer equations than unknowns there is never exactly one answer: there are infinitely many, or none. One has to be chosen, and a natural choice is the smallest. But the size of a vector can be measured in more than one way, and the ways disagree: ordinary length usually picks an answer spread over all the coordinates, while the sum of the sizes of the coordinates finds a sparse one, with many coordinates exactly 0. You need lines, planes and lengths.

# The mathematics

## Why so many answers

If a point satisfies the equations, so does that point plus any arrow that makes every left-hand side 0. For $x + 2y = 2$ that is any multiple of $(2, -1)$, since $2 + 2 \cdot (-1) = 0$. With fewer equations than unknowns such arrows always exist, so one answer brings at least a whole line of them. There are none when the equations contradict each other, like $x + y = 1$ and $x + y = 2$.

## Ways to measure size

The $p$-norm of $(x, y)$ is

$$ \lVert (x, y) \rVert_p = \left( |x|^p + |y|^p \right)^{1/p} $$

- $p = 2$, the 2-norm, is the ordinary length, $\sqrt{x^2 + y^2}$.
- $p = 1$, the 1-norm, is $|x| + |y|$.
- As $p$ grows it comes to the larger of $|x|$ and $|y|$: the max-norm, $p = \infty$.

The ball of a norm is every point of size at most some $r$: a disc for $p = 2$, a diamond with its corners on the axes for $p = 1$, a square for $p = \infty$.

## Grow the ball until it touches

Grow the ball from nothing: the first point of the line it touches is the smallest answer in that norm.

- The disc touches at the foot of the perpendicular from $0$. For one equation $a_1 x + a_2 y = b$ that is $b\,(a_1, a_2) / (a_1^2 + a_2^2)$: the shortest answer, which a formula called the pseudo-inverse gives in general.
- Along the line, $|x| + |y|$ changes at a steady rate except where the line crosses an axis. So its smallest value is at one of those crossings, where a coordinate is 0: the diamond touches with a corner. If the line runs parallel to an edge of the diamond, the whole edge touches at once, a tie, and its two ends are still sparse.
- The square, unless the line is parallel to one of its edges, touches with a corner, where $|x| = |y|$: the max-norm spreads the answer out.

In more unknowns, with $m$ equations that have an answer, a smallest answer in the 1-norm can always be found with at most $m$ coordinates that are not 0. Below $p = 1$ the ball is pinched in between spikes on the axes; its answers are sparse too, but with many unknowns no quick way is known to find them.

## An example

For $x + 2y = 2$:

- 2-norm: $\tfrac{2}{5}(1, 2) = (0.4, 0.8)$, of length $\sqrt{0.8} \approx 0.894$. Its $|x| + |y|$ is 1.2.
- 1-norm: the line crosses the axes at $(2, 0)$, of size 2, and $(0, 1)$, of size 1. So the answer is $(0, 1)$.
- max-norm: $x = y$ gives $3x = 2$, so $(2/3, 2/3)$, of size about 0.667.

# In the picture

- The blue line is every answer of the equation; drag its dots.
- The filled ball is every point up to the size shown; the white dot is where it first touches the line.
- Pink is the 1-norm (diamond, or octahedron in space), cyan the 2-norm (disc or sphere) and yellow the max-norm (square or cube).
- Green in the table marks a coordinate that is exactly 0.

> Try first "the sparsest answer": the diamond first touches the line at a corner, $(0, 1)$, where $x$ is exactly 0.

see: subspaces, overdet, sensing, sae
