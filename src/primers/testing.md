# The idea

A hypothesis test asks whether data could easily have come about if nothing were going on. It assumes, for the sake of argument, that nothing is, works out where a number made from the data would then fall, and measures how surprising the observed one is: the $p$-value. Acting on it can go wrong in two ways. It builds on the scenes on estimators and confidence intervals.

# The mathematics

## The null hypothesis and the statistic

The data are $n$ points from a bell of mean $\mu$ and spread $\sigma$. The null hypothesis says nothing is going on: $\mu = \mu_0$, here $\mu_0 = 0$. The alternative, fixed before looking, is $\mu \ne \mu_0$ (two tails) or $\mu > \mu_0$ (one tail). The test statistic is the distance of the sample's mean $\bar x$ from $\mu_0$ in standard errors:

$$ z = \frac{\bar x - \mu_0}{\sigma / \sqrt{n}} $$

Since the points come from a bell, $\bar x$ is normal about $\mu$ with spread $\sigma/\sqrt{n}$; so if the null is true, $z$ is standard normal: the purple curve. With $\sigma$ estimated by the sample's $s$, the statistic is $t$ and the curve Student's $t$, with $n - 1$ degrees of freedom.

## The $p$-value, and what it is not

With $Z$ a number drawn from the purple curve,

$$ p = P(|Z| \ge |z|) \quad \text{(two tails)}, \qquad p = P(Z \ge z) \quad \text{(one tail)} $$

the pink area: the chance, if the null is true, of a statistic at least as far out as the observed one. A level $\alpha$ is fixed beforehand, and the null is rejected when $p \le \alpha$: when $z$ lies beyond the critical value, 1.96 either side for two tails at 0.05.

Example, the opening sample: $n = 20$, $\sigma = 1$, $\bar x = 0.257$. The standard error is $1/\sqrt{20} = 0.224$, so $z = 0.257 / 0.224 = 1.15$ and $p = P(|Z| \ge 1.15) = 0.25$: not rejected. Yet the truth here is $\mu = 0.5$: a real effect, missed. A large $p$ does not show the null true.

Nor is a small $p$ the chance that the null is true: it was worked out by assuming the null true. In the last experiment one study in ten has a real effect, found with chance 0.81. Then $0.9 \times 0.05 = 0.045$ of all studies are false alarms and $0.1 \times 0.81 = 0.081$ true finds: about 36 % of results with $p \le 0.05$ come from true nulls. That share depends on how many tested hypotheses are true, which $p$ cannot know. And $p$ is not the size of an effect: with $n = 40\,000$, the scene's sample has $p < 0.0001$ for an effect of 0.02 standard deviations.

## Two kinds of error, and power

A false alarm (error of the first kind) rejects a true null; its chance is $\alpha$, since that is how the critical values were placed. A miss (error of the second kind) keeps the null when the effect is real; its chance is $\beta$, and the power is $1 - \beta$. When the effect is real, $z$ is normal with spread 1 about

$$ \delta = \frac{(\mu - \mu_0)\sqrt{n}}{\sigma} $$

the green curve, and the power is its share beyond the critical values. Above, $\delta = 0.5\sqrt{20} = 2.24$, and the power is about $P(Z \ge 1.96 - 2.24) = 0.61$: four studies in ten miss. Power grows with the effect (on the side the test looks at), with $n$ and with $\alpha$; it falls as $\sigma$ grows.

# In the picture

- Cyan dots: the sample. Yellow: its mean $\bar x$, and $z$ on the chart (drag the dot under it).
- Purple: $\mu_0$ and the null's curve; its shaded tails beyond the dashed critical values hold $\alpha$.
- Pink: the $p$-value.
- Green: the true $\mu$, and the curve about $\delta$ where $z$ really falls; shaded, the power.
- Outlined bars and the table: many studies, and the share that rejected.

> Try first "nothing going on": this sample's $p$ is 0.015, a false alarm, while the share of studies that reject settles near $\alpha$.

see: estimators, intervals, pvalues, twogroups
