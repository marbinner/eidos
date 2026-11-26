# The idea

A sample is $n$ numbers drawn from a distribution that is not known. Before estimating from it, look at it. This scene shows one sample six ways along one axis: what each picture shows, hides or makes up. You need densities, the mean and the standard deviation, from the scenes before.

# The mathematics

## Pictures that need a width

A histogram cuts the axis into bins of width $w$, with a bar over each as high as the share of the points in it divided by $w$, so that the bars' areas add up to 1:

$$ \text{height} = \frac{\text{points in the bin}}{n\,w} $$

The smoothed curve, a kernel density estimate, puts a small bell of spread $h$ (the bandwidth) on every point $x_i$ and takes the mean of these bumps. With $\varphi$ the density of the standard bell:

$$ \hat f(x) = \frac{1}{n} \sum_{i=1}^{n} \frac{1}{h}\, \varphi\left(\frac{x - x_i}{h}\right) $$

Both estimate the density $f$ the points came from. Here $f$ is known, so each picture's error can be measured: the squared gap, added up along the axis.

$$ \int (\hat f - f)^2\,dx $$

Too wide a $w$ or $h$ smears features away (two humps become one); too narrow follows this sample's accidents (a comb of peaks). The best width lies between and depends on the unknown $f$, so a rule of thumb can miss: the scene's rule for $h$, made for a bell, smooths two humps too much. Even where the bins start can turn one heap into two humps.

## Pictures that need no width

- The cumulative steps: the share of the points at or below each value, rising by $1/n$ at each point.
- The box plot: a box from the lower quartile $Q_1$ to the upper $Q_3$ (a quarter and three quarters of the way up the sorted points), a line at the median, and whiskers to the furthest points within $1.5\,\mathrm{IQR}$ of the box, where $\mathrm{IQR} = Q_3 - Q_1$. A few numbers make the whole plot, so samples of different shapes can share one.
- The quantile plot: the $i$-th smallest point goes across at its value and up at its normal score, the value below which a standard bell has a share $(i - \tfrac12)/n$. Points from a bell of mean $\mu$ and spread $\sigma$ lie near $\mu + \sigma$ times their scores, so they fall near a straight line; a bend says how the sample differs from a bell.

## Summaries, and what moves them

$$ \bar x = \frac{1}{n}\sum x_i, \qquad s^2 = \frac{1}{n-1}\sum (x_i - \bar x)^2 $$

The mean $\bar x$ and the standard deviation $s$ use every value at its size, so one wild value drags them. The median and the quartiles depend only on the values at set places in the order, so a wild value hardly moves them: they are robust.

Example: the points 1, 2, 3, 4 and 10. The mean is 4, the median 3, and $s^2 = (9 + 4 + 1 + 0 + 36)/4 = 12.5$, so $s \approx 3.54$. The quartiles, as the scene counts them, are the 2nd and 4th points, 2 and 4, so $\mathrm{IQR} = 2$; 10 lies beyond $4 + 1.5 \times 2 = 7$, so it is drawn singly. Change 10 to 5: the mean falls to 3, the median stays at 3. The smallest point's normal score is the value with a share 0.1 of a standard bell below it: $-1.28$.

# In the picture

- Cyan: the sample (drag a dot).
- Green: the true density $f$, its cumulative curve, and where its points would lie in the quantile plot.
- Yellow: the smoothed curve $\hat f$; and $\bar x \pm s$ under the box.
- Orange: the box (median and quartiles), and the line through the quartiles in the quantile plot.
- Pink: points beyond the whiskers; and, in a corner of each density chart, its error against its width.
- White: the handles that set $w$, the bins' start and $h$; and the point under the pointer.

> Try first "too wide hides", then "too narrow invents": the same 200 points as one hump, two, or a comb.

see: distributions, moments, estimators
