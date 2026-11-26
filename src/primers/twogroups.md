# The idea

Two groups of measurements, $A$ and $B$, and a difference between their means. Would chance alone often give a difference this large if the groups were really alike? The scene asks in three ways, each assuming something different: the $t$ test, shuffling the labels, and ranks. Then: how large is the difference, and what if units are measured twice? It builds on the scene on hypothesis tests.

# The mathematics

## The $t$ test

A group has a mean $\bar x$, a standard deviation $s$ and a size $n$. A group's mean wanders from sample to sample with variance about $s^2/n$, and a difference of independent means has the sum of their variances. The test divides the difference by the resulting standard error:

$$ t = \frac{\bar x_B - \bar x_A}{\sqrt{s_A^2 / n_A + s_B^2 / n_B}} $$

If the true means are equal and both groups are bell-shaped, $t$ closely follows Student's $t$, on degrees of freedom worked out from the spreads and sizes: Welch's test, drawn here. For other shapes it does so once the groups are large. The classical pooled test assumes equal true spreads, and uses one spread from both groups,

$$ s_p^2 = \frac{(n_A - 1)\,s_A^2 + (n_B - 1)\,s_B^2}{n_A + n_B - 2} $$

Exact for bell-shaped groups with equal spreads, it can be badly wrong when spreads and sizes both differ.

Example, the opening data: 20 points a group, means 1.045 apart, spreads 1.06 and 1.07. The standard error is $\sqrt{1.06^2/20 + 1.07^2/20} \approx 0.34$, so $t \approx 3.1$ (unrounded: $t = 3.11$, $p = 0.0036$).

## Shuffling, and ranks

If the labels meant nothing, every way of dealing the values into groups of these sizes was as likely as the real one. So deal them again many times: the $p$-value is the share of deals, the real one among them, whose difference is at least as far from 0. Take $A = \{1, 2, 4\}$ and $B = \{3, 5, 6\}$, a difference of $14/3 - 7/3 = 2.33$. Of all $\binom{6}{3} = 20$ deals, 4 give a difference at least this far from 0 ($B$ being $\{3, 5, 6\}$, $\{4, 5, 6\}$, $\{1, 2, 3\}$ or $\{1, 2, 4\}$), so $p = 4/20 = 0.2$. This assumes no shape, only that the groups would then be alike in every way, spreads too.

The rank test (Mann–Whitney) counts $U$, the pairs (one value from each group) in which $B$'s is larger. Above, $B$'s values beat 2, 3 and 3 of $A$'s: $U = 8$ of 9 pairs. Only order counts: moving the largest value further out changes nothing.

## How large, and pairs

The size of the difference is Cohen's

$$ d = \frac{\bar x_B - \bar x_A}{s_p} $$

the difference in standard deviations. For two groups of $n$ each the pooled $t$ is $d\sqrt{n/2}$: $t$ grows with $n$, and $d$ does not, so a small $p$ may come from a large $d$ or from a large $n$. For two bells of one spread, $B$'s value is the larger in a share $\Phi(d/\sqrt{2})$ of the pairs, with $\Phi$ the standard normal's cumulative probability: only 76 % at $d = 1$, so such groups overlap a great deal.

With each unit measured twice, the paired test tests the mean of the units' differences $B - A$ against 0 ($t$ on $n - 1$ degrees of freedom). Much of the spread from unit to unit cancels in each difference: with equal spreads $\sigma$ and correlation $\rho$ within a unit, $B - A$ has variance $2\sigma^2(1 - \rho)$, a tenth of $2\sigma^2$ when $\rho = 0.9$.

# In the picture

- Cyan: group $A$, with its mean (the tall tick) and its bell. Orange: group $B$.
- Yellow: the difference of the means (the bracket), and where it falls among the differences chance gives.
- Purple: what chance gives if the labels mean nothing: Student's $t$ (the curve) and the shuffles (the bars).
- Pink: as far from 0 as the real difference, or further: the $p$-value.

> Try first "a clear difference": $t = 3.11$ and the shuffles agree, yet $B$'s value is the larger in only 75.5 % of pairs.

see: testing, bootstrap, anova, confound
