# The idea

An equation in two unknowns, like $0.6x + 0.8y = 2$, holds on a line. Two lines usually cross at one point; a third line usually misses it, and then no point satisfies all the equations. With more equations than unknowns, the best one can do is the point that misses the lines least. But "least" can be measured in more than one way, and each way can give a different point with a character of its own. You need lines and the dot product.

# The mathematics

## How much a point misses

For the equation $a_1 x + a_2 y = b$, the residual of the point $(x, y)$ is

$$ r = a_1 x + a_2 y - b $$

It is 0 on the line, positive on one side and negative on the other. Each equation here is scaled so that $(a_1, a_2)$ has length 1; then $|r|$ is the distance from the point to the line. Why: $(a_1, a_2)$ is at right angles to the line (moving along the line leaves $a_1 x + a_2 y$ as it is), and $r$ is its dot product with the step from any point of the line to $(x, y)$: the length, with a sign, of that step's shadow across the line.

## Three ways to total the misses

$$ \sum r^2, \qquad \sum |r|, \qquad \max |r| $$

Least squares, the first, is the usual choice because it has a formula. At its best point

$$ \sum r\,(a_1, a_2) = (0, 0) $$

which are the normal equations $A^\top A x = A^\top b$, where $A$ has a row $(a_1, a_2)$ for each equation and $A^\top$ is $A$ with rows and columns swapped. Why: a short step $d$ changes each $r$ by $(a_1, a_2) \cdot d$, so the total of squares changes by about $2\, d \cdot \sum r\,(a_1, a_2)$; if that sum were not $(0, 0)$, a step against it would lower the total. Squares make a large miss count heavily, so one wild equation can pull the point a long way.

Least absolute values, the second, has a best point where two lines cross (unless all the lines are parallel): between the lines the total changes at a steady rate along a straight path, so the point can slide without making it worse until it meets a line, then along that line until it meets another. Sliding a far line further away does not move it.

Least maximum, the third, makes the largest miss as small as it can be. Its best point is usually equally far from the three worst lines; for three lines that make a triangle, it is the centre of the largest circle inside it.

## An example

Take the lines $x = 0$, $y = 0$ and $0.6x + 0.8y = 1$; each $(a_1, a_2)$ has length 1.

- Least squares: $(0.3, 0.4)$. The residuals are $0.3$, $0.4$ and $-0.5$, and $0.3\,(1, 0) + 0.4\,(0, 1) - 0.5\,(0.6, 0.8) = (0, 0)$. The total of squares is 0.5.
- Least absolute values: $(0, 0)$, on the first two lines, missing the third by 1. The other crossings do worse: $5/3$ at $(5/3, 0)$, 1.25 at $(0, 1.25)$.
- Least maximum: $(5/12, 5/12)$, about $(0.417, 0.417)$, which misses all three lines by $5/12$: for the third, $0.6 \cdot 5/12 + 0.8 \cdot 5/12 - 1 = -5/12$.

With two nearly parallel lines, a small change in a $b$ moves their crossing a long way: the system is ill-conditioned.

# In the picture

- The blue lines are the equations, numbered as in the table. A line's ringed dot slides it; its dot with arrows turns it.
- The white dot is a point to try; pink, how far it misses each line (for $\sum r^2$ a square on each miss).
- The cyan, orange and green points are the best for least squares, least absolute values and least maximum.
- The faint curves join points with the same total.

> Try first "a third equation: no answer": no point is on all three lines, and the three best points part.

see: elim, lsq, regress, condition
