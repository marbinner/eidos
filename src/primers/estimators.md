# The idea

An estimate is a number worked out from a sample to stand for something about the population it came from, such as its variance. Another sample gives another estimate, so an estimate has a distribution of its own. Here the population is known, and three rules for the same quantity are judged by how their estimates from thousands of samples fall around the truth. You need the mean and the variance, and how averages behave, from the scenes before.

# The mathematics

## An estimator and its two errors

An estimator $T$ is a rule applied to a sample; an estimate is what it gives for one sample. It aims at a true value, such as $\sigma^2$, the population's variance. Over all the samples that could be drawn, its estimates have a distribution, the sampling distribution; its standard deviation is the standard error. The bias is how far off the estimates are on average:

$$ \text{bias} = \text{average of } T - \text{truth} $$

The mean squared error, the average of $(T - \text{truth})^2$, splits into the two ways of being wrong:

$$ \text{mean squared error} = \text{bias}^2 + \text{variance} $$

Why: each estimate's miss is its distance from the average of the estimates, plus the bias. Square and average: the first part gives the variance, the second the bias squared, and the cross term averages to 0, since the first part does.

For the mean $\bar x$ of $n$ independent values, variances add, so $\bar x$ has variance $\sigma^2/n$ and standard error $\sigma/\sqrt{n}$: four times the data halves it.

## Why divide by $n - 1$

Let $S = \sum (x_i - \bar x)^2$, the squared distances from the sample's own mean $\bar x$. That mean is the point closest to the sample (in summed squared distance), so the same sum measured from the true mean $\mu$ is never smaller:

$$ \sum (x_i - \mu)^2 = S + n(\bar x - \mu)^2 $$

Average over samples: the left side averages $n\sigma^2$, each term $\sigma^2$. The last term averages $n \cdot \sigma^2/n = \sigma^2$. So $S$ averages $(n - 1)\sigma^2$: $S/(n - 1)$ is unbiased, and $S/n$ comes out low by the factor $(n - 1)/n$. This needs independent values and a finite variance, not a bell.

Example: the values 1, 2 and 6, from a population with $\mu = 2$. Then $\bar x = 3$ and $S = 4 + 1 + 9 = 14$; from $\mu$ the sum is $1 + 0 + 16 = 17$, which is $14 + 3 \times (3 - 2)^2$. The three estimates of $\sigma^2$ are $S/3 \approx 4.67$, $S/2 = 7$ and $S/4 = 3.5$.

## Right on average is not best

Dividing by a larger number pulls every estimate towards 0: it adds bias and takes away wobble. From a bell with $\sigma^2 = 1$ and $n = 5$, the mean squared errors in the long run are 0.36 for $S/n$, 0.50 for $S/(n - 1)$ and 0.33 for $S/(n + 1)$: the most biased wins. Which wins depends on the population. For the centre likewise: from a bell the mean wobbles less than the median; with the heavy tails here, the median wobbles less.

An estimator is consistent when the chance that it misses the truth by more than any given distance shrinks to 0 as $n$ grows. $S/n$ is biased but consistent: its bias, $-\sigma^2/n$, dies away, and here so does its wobble.

# In the picture

- Green: the population, and the true value on each histogram.
- Cyan: the latest sample; white: its mean $\bar x$ and each value's distance from it.
- Yellow, orange and blue: the three estimators, for the spread $S/n$, $S/(n - 1)$ and $S/(n + 1)$. Each histogram, of the estimates so far, approaches the sampling distribution; its line is the average of the estimates, the bar under it one standard error either side.
- Pink: the bias, from the truth to the average; and $\text{bias}^2$, the first part of each bar of mean squared error.

> Try first "dividing by $n$ comes out low", and compare the three bars of error.

see: moments, clt, likelihood, intervals
