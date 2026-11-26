# The idea

Often a few numbers must stand for a whole distribution: its centre, its spread, its lean. Here the shape rests on a beam, its mean the balance point, and a sample under it shows what the mean and the median each answer to. It follows the scene on distributions.

# The mathematics

## Three centres

A sample of $n$ numbers $x_1, \dots, x_n$ has the mean

$$ \bar x = \frac{x_1 + x_2 + \dots + x_n}{n} $$

For a distribution with density $f$, the mean $\mu$ is the average weighted by probability (the integral is that weighted sum):

$$ \mu = \int x\, f(x)\, dx $$

The median has half the values (or half the area) on each side; the mode is where the shape peaks.

## The balance point

Rest the shape on a beam over a pivot at $p$. Each bit of area turns the beam by its size times its distance from the pivot; each side's total is its turning moment. The right one less the left one is the mean of $x - p$, which is $\mu - p$, so the beam is level only with the pivot at the mean. A little area far out turns the beam as hard as much area close in: a long tail can pull the mean far from the peak and the median. With tails heavy enough, both turning moments are infinite: no mean (the Cauchy).

## Spread

The variance $\sigma^2$ is the mean squared distance from the mean, and the standard deviation $\sigma$ its square root, in the data's own units. For a pivot at $p$, the average square splits in two ($\sum$ adds over $i$ from 1 to $n$):

$$ \frac{1}{n}\sum_{i=1}^{n} (x_i - p)^2 = \frac{1}{n}\sum_{i=1}^{n} (x_i - \bar x)^2 + (\bar x - p)^2 $$

Why: each $x_i - p$ is $(x_i - \bar x) + (\bar x - p)$; squared and added, the cross terms cancel, since the $x_i - \bar x$ add up to 0. So the average square is least with the pivot at the mean, where it is the variance (dividing by $n$). For a distribution it is $\sigma^2 + (\mu - p)^2$. The average distance, unsquared, is least at the median instead: moving the pivot towards the side with more points shortens more distances than it lengthens.

Example: the sample 1, 2, 6 has mean 3 and median 2. The average square is $(4 + 1 + 9)/3 = 14/3$ at 3, and $(1 + 0 + 16)/3 = 17/3$ at 2, which is $14/3 + 1^2$. The average distance is $(1 + 0 + 4)/3 = 5/3$ at 2, against 2 at 3. Move the 6 to 30: the mean jumps to 11; the median stays.

## Shape, and a bound

The means of $(x - \mu)^2$, $(x - \mu)^3$ and $(x - \mu)^4$ are the second, third and fourth moments about the mean, when finite. The third over $\sigma^3$ is the skewness: 0 for a symmetric shape, and as a rule positive for a long right tail. The fourth over $\sigma^4$, less 3, is the excess kurtosis: 0 for the bell and, again as a rule, positive for heavy tails.

Chebyshev's inequality: if the variance is finite, at most $1/k^2$ of the distribution lies $k$ or more standard deviations from the mean. Why: each value out there adds at least $k^2 \sigma^2$ to the mean square, so a share $q$ of them adds at least $q k^2 \sigma^2$, which cannot exceed $\sigma^2$. At $k = 2$ that is 25 %; the bell has 4.6 %.

# In the picture

- Green: the density $f$ on the beam, with its mean $\mu$ (the triangle), median (dashed) and mode (the dot).
- White: the pivot $p$ (drag it), and the bar of one $\sigma$ each side of the mean.
- Cyan dots: a sample. Yellow: the squares on their distances from the pivot, and the sample mean; orange: the distances, and the sample median.
- The lowest chart: the average square against $p$.
- Pink: what lies beyond $k$ standard deviations.

> Try first "the balance point", and drag the pivot until the beam is level.

see: distributions, lln, describe, estimators
