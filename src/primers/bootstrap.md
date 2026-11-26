# The idea

How much would a statistic, such as a mean or a median, change if the study were done again with a new sample? The scene on estimators answered by drawing new samples from a known population. In practice there is one sample, and the population is unknown. The bootstrap uses the sample in place of the population and draws new samples from the sample itself. It needs no formula, so it works for statistics that have none, when the sample resembles the population in what the statistic depends on. You need the sampling distribution and the standard error (estimators) and confidence intervals.

# The mathematics

## Resampling

A resample is $n$ draws from the sample's $n$ points, each draw at random from all $n$ of them, with replacement: each point drawn is put back, so it can be drawn again. Some points come twice or more, some not at all. Work the statistic out for each of thousands of resamples. Their values form the bootstrap distribution, and

$$ \text{SE} \approx \text{the standard deviation of the resamples' values} $$

while their middle 95 %, from the 2.5th to the 97.5th percentile, is the percentile interval for the true value.

Why it can work: the sample stands to its resamples as the population stands to its samples. If the sample looks like the population, a statistic wobbles across resamples, round the sample's value, about as it wobbles across fresh samples round the true value. So the bootstrap distribution is centred near the sample's value, not on the truth: it shows how far an estimate wanders, not where the truth is.

## Who is in a resample

One draw misses a given point with chance $1 - 1/n$, and all $n$ draws miss it with chance $(1 - 1/n)^n$. So on average a resample holds a share

$$ 1 - (1 - 1/n)^n $$

of the points: 64.8 % for $n = 12$, and close to $1 - 1/e$, or 63.2 %, for large $n$.

## A check by hand

Take the sample 0 and 1, so $n = 2$. Its four resamples, equally likely, are $(0, 0)$, $(0, 1)$, $(1, 0)$ and $(1, 1)$. They hold 1, 2, 2 and 1 of the points: 75 % on average, which is $1 - (1/2)^2$. Their means are 0, 0.5, 0.5 and 1, whose standard deviation is $\sqrt{(0.25 + 0 + 0 + 0.25)/4} = 0.354$. The formula $s/\sqrt{n}$ gives $0.707/\sqrt{2} = 0.5$. The two differ by the factor $\sqrt{(n - 1)/n}$, here $\sqrt{1/2}$: the bootstrap treats the sample as the whole population, whose variance divides by $n$, not $n - 1$. At $n = 30$ the factor is $\sqrt{29/30} = 0.983$: that is why, in "the mean", the bootstrap's 0.177 falls a shade below the formula's 0.180.

## Where it fails

- A statistic that hangs on the sample's edge. A resample's largest value can never be larger than the sample's largest; a fresh sample's often is.
- Too little data. With 8 values the percentile interval for the mean is too narrow (it held the true mean in 881 of 1,000 new samples): it divides by $n$, and nothing allows for $s$ being a guess, as Student's $t$ does.
- A median. With an odd $n$, a resample's median is always one of the sample's own values, so the bars come in spikes, and the standard error found is rough.

# In the picture

- Green: the population, the true value, and the statistic of 10,000 fresh samples: the truth the bootstrap tries to find.
- Cyan: the sample's points, and their copies in the resample being made; dim, the points it left out.
- Yellow: the resample's statistic, the bars of all the resamples' values, and their middle 95 %.
- White: the sample's own statistic.

> Try first "who is in a resample", then "the mean", and compare the yellow bars with the green outline.

see: estimators, intervals, describe, twogroups
