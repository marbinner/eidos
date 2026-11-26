# The idea

A cloud of data points, each three numbers $x$, $y$ and $z$, has a shape: it may be spread mostly along a line, or over a plane like a pancake. Principal components are the directions of that shape: PC1, the direction of most spread; PC2, the most spread at right angles to PC1; PC3 at right angles to both. Flattening the cloud onto the first one or two describes it with fewer numbers, losing as little as possible. You need quadratic forms, and shadows on a plane from least squares.

# The mathematics

## Spread along a direction

Measure each point from the mean (the average of all $n$ points) and call it $p$. Its coordinate along a unit direction $u$ is $u \cdot p$. The variance along $u$ is the sum of $(u \cdot p)^2$ over the points, divided by $n - 1$; its square root is the spread. Since $(u \cdot p)^2 = u^\top (p\, p^\top) u$, where $p\, p^\top$ is a column times a row, the sum is a quadratic form:

$$ \text{variance along } u = u^\top C u, \qquad C = \frac{1}{n - 1} \sum p\, p^\top $$

$C$ is the covariance matrix: the variances of $x$, $y$ and $z$ on its diagonal, and off it how two coordinates vary together. It is symmetric, so $u^\top C u$ is largest along the eigenvector of its largest eigenvalue: that is PC1, and the eigenvalue $\lambda_1$ is the variance along it. PC2 and PC3 are the other eigenvectors, at right angles, with $\lambda_1 \ge \lambda_2 \ge \lambda_3$. A component's share of the variance is its $\lambda$ over $\lambda_1 + \lambda_2 + \lambda_3$.

## Most spread kept is least lost

Drop each point straight onto the plane of PC1 and PC2: there is its shadow, and the step between them, at right angles to the plane, is its miss. By Pythagoras' theorem, with distances from the mean,

$$ |p|^2 = |\text{shadow}|^2 + |\text{miss}|^2 $$

The left side, added up over the points, is the same for every plane through the mean, so keeping the most variance is losing the least sum of squared misses. The plane of PC1 and PC2 keeps the most, so no plane loses less; the share lost is $\lambda_3 / (\lambda_1 + \lambda_2 + \lambda_3)$.

Example, four points in the plane: $(3, 3)$, $(-3, -3)$, $(1, -1)$ and $(-1, 1)$, with mean $(0, 0)$ and $n - 1 = 3$:

$$ C = \frac{1}{3} \begin{bmatrix} 20 & 16 \\ 16 & 20 \end{bmatrix} $$

($20 = 9 + 9 + 1 + 1$ and $16 = 9 + 9 - 1 - 1$). $C$ sends $(1, 1)$ to $12 \cdot (1, 1)$ and $(1, -1)$ to $(4/3) \cdot (1, -1)$, so PC1 lies along $(1, 1)$ with $\lambda_1 = 12$, and $\lambda_2 = 4/3$. PC1 keeps $12 / (12 + 4/3) = 90\%$. Flattened onto its line, the last two points miss by $\sqrt{2}$: squared misses $2 + 2 = 4$, out of squared distances $18 + 18 + 2 + 2 = 40$, the 10% lost.

## To keep in mind

- Components depend on the units: in "change the units", PC1 swings round by 50°. So coordinates are often first divided by their spread.
- Misses count squared, so one far point can tilt the plane.
- PC1 is not the regression line of $y$ on $x$, which has the least squared misses measured up and down, not at right angles.
- With the points $p$ as the rows of a matrix $X$, $X^\top X = (n - 1)C$: the components are $X$'s right singular vectors, and $\sigma^2 / (n - 1) = \lambda$.

# In the picture

- Dots: the data; white dot: the mean.
- Orange, purple and green arrows: PC1, PC2 and PC3, each as long as the spread along it, $\sqrt{\lambda}$.
- Blue: the plane (or line) kept; pink: misses.
- The bars: each component's share. The PCA plot: the points at their coordinates along PC1 and PC2.
- In the plane, cyan: the regression line. Red: a far point, when there is one.

> Try first "flatten a pancake": each point drops along its pink miss onto the blue plane.

see: quadform, lsq, lowrank, gauss2
