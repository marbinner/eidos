# The idea

An estimate such as a sample's mean comes without a word on how far off it may be. A confidence interval adds a range, made by a recipe that, used on sample after sample, covers the true value in a stated share of them. Here the truth is known, so you can watch interval after interval cover it or miss, and see exactly what the 95 % does and does not promise. You need the standard error, from the scene on estimators, and the bell shape that means take, from the central limit theorem.

# The mathematics

## Where the recipe comes from

Take $n$ values from a bell with mean $\mu$ and standard deviation $\sigma$, with $\sigma$ known. Their mean $\bar x$ varies from sample to sample with standard error $\sigma/\sqrt{n}$, as a bell centred on $\mu$ (exactly for a bell; for other populations nearly so when $n$ is large). A bell holds 95 % of its probability within 1.96 standard deviations of its centre, so before the sample is drawn:

$$ P\left(|\bar x - \mu| \le 1.96\,\sigma/\sqrt{n}\right) = 0.95 $$

"$\bar x$ lands within $1.96\,\sigma/\sqrt{n}$ of $\mu$" is the same event as "$\mu$ lies within $1.96\,\sigma/\sqrt{n}$ of $\bar x$". So the interval $\bar x \pm 1.96\,\sigma/\sqrt{n}$ covers $\mu$ in 95 % of samples. Every recipe here has this shape:

$$ \text{estimate} \pm \text{critical value} \times \text{standard error} $$

The critical value sets the level (1.96 for 95 %, 2.58 for 99 %); the standard error sets the scale, so four times the data halves the width.

Example: $\sigma = 2$ and $n = 10$ give $\text{SE} = 2/\sqrt{10} = 0.632$, and $1.96 \times 0.632 = 1.24$. A sample with mean 9.83 gives $[8.59, 11.07]$, which covers $\mu = 10$.

## When $\sigma$ is not known

Put the sample's $s$ in place of $\sigma$. With few values $s$ is often below $\sigma$, and the interval is then narrower than $\sigma$ would make it. For values from a bell, $(\bar x - \mu)/(s/\sqrt{n})$ follows Student's $t$ distribution with $n - 1$ degrees of freedom, whose tails are heavier than the bell's; at $n = 5$ its critical value for 95 % is 2.78, not 1.96. With 1.96 and $s$ at $n = 5$, only about 88 % cover.

## What 95 % promises, and what it does not

The chance is in the recipe, not in $\mu$: $\mu$ is a fixed number, and the interval is what varies.

- It promises that, before the sample is drawn, the interval to come has a 95 % chance of covering $\mu$, whatever $\mu$ is; so over many samples the share that cover settles near 95 %.
- It does not say that this interval, $[8.59, 11.07]$, holds $\mu$ with probability 95 %. Once the sample is drawn, the interval covers $\mu$ or it does not, and without the truth nothing tells which.
- It says nothing about where 95 % of the values lie.
- It does not promise 95 of any given 100: exactly 95 happens in only about 18 % of runs of 100.
- It holds only when the recipe's assumptions do. With ten values from a long right tail, Student's $t$ covers about 84 %.

A probability for $\mu$ itself needs a prior belief about $\mu$. Given one, a credible interval does say "$\mu$ is in here with probability 95 %", as far as that prior and the model are right (the scene on Bayesian updating).

# In the picture

- Green: the truth, $\mu$ or $p$, and the population.
- Yellow: an interval that covers the truth; its tick is the sample's estimate.
- Pink: an interval that misses, wholly below or wholly above.
- Blue: an interval while the truth is hidden.
- Cyan and the white band: one sample and its interval, taken apart.
- Dashed line: the promised level, against the share that has covered so far.

> Try first "a hundred intervals", then "what 95 % does not mean".

see: clt, estimators, bootstrap, bayes
