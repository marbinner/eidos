# The idea

Two measurements of the same things often rise together. The correlation coefficient $r$ says how closely their scatter follows a straight line. The scene shows how it is worked out, why the best line predicting one from the other has slope $r$ in standard units, why extremes are followed by less extreme values, and how $r$ misleads. You need the mean and standard deviation (scene moments).

# The mathematics

## Standard units and $r$

There are $n$ points $(x_i, y_i)$. Write $\bar x$ for the mean of the $x_i$ and $s_x$ for their standard deviation, dividing by $n$; likewise $\bar y$ and $s_y$. In standard units a value says how many standard deviations it lies from its mean, and $r$ is the average product:

$$ \tilde x_i = \frac{x_i - \bar x}{s_x}, \qquad \tilde y_i = \frac{y_i - \bar y}{s_y}, \qquad r = \frac{1}{n} \sum_{i=1}^{n} \tilde x_i \, \tilde y_i $$

Points above both means, or below both, add to $r$; those above one and below the other take away.

Why $-1 \le r \le 1$: each variable's squares average 1 in standard units, so the average of $(\tilde x_i - \tilde y_i)^2$ is $2(1 - r)$. Squares cannot average below 0, so $r \le 1$, and $r = 1$ only if $\tilde x_i = \tilde y_i$ for every point: all on one rising line. With $\tilde x_i + \tilde y_i$ instead, $r \ge -1$.

## The regression line

Predict $\tilde y$ by $m \tilde x$, a line through the centre (moving it up by $c$ only adds $c^2$). Its average squared miss is

$$ \frac{1}{n} \sum_{i=1}^{n} (\tilde y_i - m \tilde x_i)^2 = 1 - 2mr + m^2 = (1 - r^2) + (m - r)^2 $$

smallest at $m = r$. So in standard units the regression line of $y$ on $x$ has slope $r$, and it accounts for the share $r^2$ of the variance of $y$, leaving $1 - r^2$. In the data's own units its slope is $r\, s_y / s_x$, through $(\bar x, \bar y)$. Predicting $x$ from $y$ gives $\tilde x = r \tilde y$, of slope $1/r$ drawn with $x$ across: another line, unless $r = \pm 1$. The SD line, slope $\pm 1$ in standard units, lies between them.

Example: the points $(1, 2)$, $(2, 1)$, $(3, 4)$, $(4, 3)$, $(5, 5)$. Both means are 3, and both standard deviations $\sqrt 2$: the squared distances 4, 1, 0, 1, 4 average 2. The products $(x_i - 3)(y_i - 3)/2$ are 1, 1, 0, 0, 2, so $r = 0.8$. The line $y = 0.6 + 0.8x$ misses by $0.6$, $-1.2$, $1$, $-0.8$, $0.4$. Their squares add to $3.6$: $1 - 0.8^2$ of 10, the sum of the $(y_i - 3)^2$.

## Regression to the mean

The point at $x = 5$ is $1.41$ standard deviations above the mean of $x$; the line puts it only $0.8 \times 1.41 = 1.13$ above the mean of $y$. Points chosen for an extreme $x$ are predicted $r$ times as far out on $y$; if they are many and the points scatter alike along the line, their average $y$ is near that. Nothing pulls them back: it is what $|r| < 1$ means.

## The traps

- $r$ sees only straight lines: $y = x^2$ has $r = 0$ for $x$ spread evenly about 0.
- One far point can make a correlation or hide one.
- Keeping only the larger $x$ of a normal cloud pulls $r$ towards 0: the line and the scatter stay, but $y$ varies less.
- A sample's $r$ varies by chance about the cloud's correlation $\rho$: from 10 points of unrelated normal variables, $|r| > 0.5$ about 14 times in 100.
- The rank correlation, $r$ of the ranks, is 1 whenever $y$ rises with every rise in $x$.

# In the picture

- Cyan dots: the data, to drag. Blue quadrants give positive products, red negative.
- Yellow: the regression line of $y$ on $x$, and $r$; orange: $x$ on $y$; purple dashes: the SD line.
- White: the top tenth on $x$, and their averages; pink: the far point.

> Try first "regression to the mean": the white points' average $y$ lies near the yellow line, short of the SD line.

see: moments, gauss2, regress, confound
