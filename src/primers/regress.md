# The idea

Points scattered about some unknown curve: what was the curve? Least squares answers by choosing, among all curves of one kind (here polynomials), the one whose misses, squared and added up, are smallest: in the picture, the curve whose pink squares have the least total area. The scene shows what happens when the curve is given too little freedom or too much, and how a penalty reins it in. You need the mean (scene moments) and the regression line (scene correlation).

# The mathematics

## The fit

There are $n$ data points $(x_i, y_i)$. The curve is a polynomial of degree $d$, written in $u$, which is $x$ rescaled to run from $-1$ at the leftmost point to $1$ at the rightmost, so that every power of $u$ stays between $-1$ and $1$ over the data:

$$ \hat y = c_0 + c_1 u + c_2 u^2 + \cdots + c_d u^d $$

At each point the curve misses by $\hat y(u_i) - y_i$, and the square drawn on that miss has the squared miss as its area. Least squares chooses the coefficients $c_0, \ldots, c_d$ that make the total area

$$ S = \sum_{i=1}^{n} \bigl(c_0 + c_1 u_i + \cdots + c_d u_i^d - y_i\bigr)^2 $$

as small as it can be. $S$ is a bowl-shaped (quadratic) function of the coefficients, so at its bottom its slope along each coefficient is 0. That gives $d + 1$ equations, each linear in the coefficients, with one solution when there are at least $d + 1$ different $x$.

## More freedom

The mean squared error on the data is $S/n$. With no penalty, raising the degree can never raise it: a curve of degree $d + 1$ can be any curve of degree $d$ (with $c_{d+1} = 0$), so its best is at least as good. With $d + 1 = n$, and the $x_i$ all different, the curve passes through every point and the error on the data is 0.

The error against the true curve is the average squared gap between the fitted curve and the true one, over $x$ from $-3.2$ to $3.2$. It usually falls while more freedom lets the curve follow the true shape, then climbs once the freedom goes into following the noise: overfitting. Degree 12 swings most at the ends.

## The ridge penalty

$$ S_\lambda = S + \lambda \left(c_1^2 + \cdots + c_d^2\right) $$

Since $u$ stays between $-1$ and $1$, wild swings need large coefficients, and the penalty makes those costly: the fit trades some area of squares for smaller coefficients. $\lambda = 0$ is plain least squares; as $\lambda$ grows, $c_1^2 + \cdots + c_d^2$ shrinks towards 0, leaving in the end the flat line at the mean of the $y_i$. The constant $c_0$ is not penalised.

## A worked example

Three points at $u = -1, 0, 1$, with $y = 1, 0, 2$. For a straight line, the $u_i$ add to 0, so the two equations separate: $c_0$ is the mean of the $y_i$, 1, and

$$ c_1 = \frac{\sum u_i y_i}{\sum u_i^2} = \frac{1}{2} $$

The line $1 + 0.5u$ misses by $-0.5$, $1$, $-0.5$, so $S = 0.25 + 1 + 0.25 = 1.5$. Degree 2 has three coefficients, as many as points: the parabola $0.5u + 1.5u^2$ goes through all three, and $S = 0$.

With a penalty $\lambda = 2$ the line's slope becomes $\sum u_i y_i / (\sum u_i^2 + \lambda) = 1/4$. Its squares grow, to $S = 1.625$, but $S_\lambda = 1.625 + 2 \times (1/4)^2 = 1.75$ is below the $1.5 + 2 \times (1/2)^2 = 2$ of the unpenalised line.

# In the picture

- White dots: the data, the true curve plus noise; drag them, or click empty space to add one.
- Grey: the true curve. Cyan: the fitted polynomial $\hat y$.
- Pink: each point's miss, and the square on it; their total area is $S$.
- The small chart: for every degree, the error on the data (cyan) and against the true curve (orange), on a log scale, with the degree in use marked white.

> Try first "too flexible", then "ridge calms it": the same degree 12, wild and then tamed.

see: correlation, lsq, modelselect
