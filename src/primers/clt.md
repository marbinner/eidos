# The idea

The law of large numbers says that an average of $n$ independent values settles on the mean, its typical distance shrinking like $\sigma/\sqrt{n}$. This scene asks what shape the averages make round the mean, and the answer is the same bell curve, the normal distribution, almost whatever the values come from. That is why the normal turns up everywhere in statistics. It leans on "Mean, spread and shape" and "The law of large numbers".

# The mathematics

## Standard units

A single value has mean $\mu$ and standard deviation $\sigma$. An average $\bar x_n$ of $n$ independent values has the same mean and a spread of $\sigma/\sqrt{n}$. To compare averages for different $n$, count each one's distance from $\mu$ in those spreads:

$$ z = \frac{\bar x_n - \mu}{\sigma / \sqrt{n}} $$

In these standard units the averages always have mean 0 and standard deviation 1. Only their shape is left to change with $n$.

## The theorem

If the values are independent draws from one distribution with a mean $\mu$ and a finite standard deviation $\sigma$ (above 0), then as $n$ grows, the probability that an average's $z$ falls in any range approaches the area over that range under the standard normal curve,

$$ \varphi(z) = \frac{1}{\sqrt{2\pi}}\, e^{-z^2/2} $$

Under this bell, 68 % of the area lies within 1 of 0, and 95 % within 1.96. The theorem does not say how large $n$ must be: that depends on the source.

## Why the shape fades

In standard units, averaging keeps the mean and the spread as they are and wears down every other feature of the shape. Take the lean, the skewness: the mean of $(x - \mu)^3$ over $\sigma^3$. For a sum of $n$ independent values, the mean cube of its distance from $n\mu$ is $n$ times one value's, as the mean square is (the cross terms average to 0), while its spread cubed, $(\sigma\sqrt{n})^3$, is $n^{3/2}$ times one value's. Dividing:

$$ \text{skewness of } \bar x_n = \frac{\text{skewness of one value}}{\sqrt{n}} $$

(if the one value's skewness is finite). An exponential value has skewness 2, so an average of 5 has skewness about 0.89, and an average of 100 has 0.2. The excess kurtosis, where it is finite, falls faster still, as $1/n$. And the bell is one shape that averaging does not change at all: the average of independent normal values is normal.

## When it fails

The Cauchy distribution has no mean and no finite spread, so the theorem does not apply. An average of $n$ Cauchy values is spread exactly as one value is, for every $n$.

## An example

A coin that shows 1 with probability 0.2, else 0, has $\mu = 0.2$. Its squared distances from 0.2 are 0.04 (for a 0) and 0.64 (for a 1), so $\sigma^2 = 0.8 \times 0.04 + 0.2 \times 0.64 = 0.16$ and $\sigma = 0.4$. An average of 100 tosses has spread $0.4/\sqrt{100} = 0.04$. So 30 ones in 100, an average of 0.3, is $(0.3 - 0.2)/0.04 = 2.5$ in standard units: far out on the bell. By the bell, about 95 % of averages lie within 1.96 standard units, from $0.2 - 0.078$ to $0.2 + 0.078$: 13 to 27 ones. The exact chance of 13 to 27, from the binomial, is 94 %.

# In the picture

- Purple, at the top: the shape each value is drawn from.
- White dots: the latest $n$ values drawn; yellow: their average falling into the bars.
- Cyan bars: where the averages landed, in standard units, scaled to a total area of 1.
- Yellow curve: the bell $\varphi$ the bars approach as $n$ grows (for the Cauchy, the shape of a single value).
- The numbers compare the averages' mean and spread with $\mu$ and $\sigma/\sqrt{n}$.

> Try first "lopsided, $n = 5$", where the bars still lean like the source; then "lopsided, $n = 100$".

see: moments, lln, intervals, gauss2
