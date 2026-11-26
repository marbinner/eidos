# The idea

A test at level $\alpha$ makes one promise: if nothing is going on, it gives a false alarm with chance $\alpha$. That promise is about one test, looked at once. Here is what becomes of it when many hypotheses are tested, many tests are done at once, or one study is looked at again and again. It builds on the hypothesis tests scene.

# The mathematics

## Why true nulls give even $p$-values

Every test here takes $n$ values, each normal with spread 1, and asks whether the mean they are drawn from is 0. With $\bar x$ their average, the statistic is $z = \bar x \sqrt{n}$, standard normal if the null is true, and

$$ p = 2\,\Phi(-|z|) $$

where $\Phi(x)$ is the share of the standard normal below $x$. Then $p \le a$ exactly when $|z|$ lies beyond the point whose two tails hold a share $a$ of the null's curve, and a true null puts $z$ there with chance $a$. So, if the null is true,

$$ P(p \le a) = a \quad \text{for every } a \text{ from 0 to 1} $$

The $p$-values of true nulls are spread evenly from 0 to 1, a twentieth in each twentieth of the range; those of real effects crowd towards 0.

## What "significant" is worth

Suppose a share $r$ of the hypotheses tested are real effects. A real one comes out significant with chance equal to the power, a true null with chance $\alpha$. Of the significant results, the share that are real is

$$ \frac{r \times \text{power}}{r \times \text{power} + (1 - r)\,\alpha} $$

Example: $r = 0.1$, power 0.8, $\alpha = 0.05$ give $0.08 / (0.08 + 0.045) = 0.64$. About a third of the discoveries are false alarms, though each has $p \le 0.05$: $p$ is not the chance that the null is true.

Significance also exaggerates. With $n = 16$ an estimate is significant only beyond $\pm 1.96/\sqrt{16} = \pm 0.49$. If the true effect is 0.25, a study is significant only when its noise pushes the estimate past one of these, nearly always past 0.49. So the significant ones overestimate the effect (in the scene they average 0.61): the winner's curse.

## Many tests at once

In a family of $m$ independent tests whose nulls are all true, the chance of at least one false alarm is

$$ 1 - (1 - \alpha)^m $$

For $m = 20$ at 0.05 that is $1 - 0.95^{20} = 0.64$. Two remedies:

- Bonferroni tests each at $\alpha/m$. The chance of any false alarm is at most the sum of the separate chances, $m \times \alpha/m = \alpha$, however the tests depend on each other. The price is power.
- Benjamini–Hochberg aims at a false discovery rate $q$: the average share of false alarms among what is declared (0 when nothing is). It sorts the $p$-values, finds the last rank $k$ whose $p$-value is at most $kq/m$, and declares that one and every smaller one. For independent tests the rate is then at most $q$.

## Peeking

A study of pure noise is tested after each new value and stopped at the first $p \le 0.05$. Each look alone has a 5 % chance, but every look is another chance: in the scene 28 % of the studies have dipped below 0.05 by the 100th value, and 40 % by the 500th. With no last look at all, every such study would dip sooner or later. The promise of $\alpha$ is for a test whose $n$ was fixed before looking.

# In the picture

- Three pictures, chosen at the top: many studies, many tests, peeking.
- Purple: true nulls. Green: real effects, and their true size.
- Pink: the level $\alpha$ (drag the pink dot) and the critical values; in a family, the false alarms; in peeking, the dips below $\alpha$.
- Yellow: the average estimate of the significant real effects.
- In a family, the white lines are the three rules' thresholds: $\alpha$, $\alpha/m$ and the line $kq/m$.

> Try first "nothing going on": ten thousand true nulls fill the twenty bars evenly, and the first bar's 5 % are all false alarms.

see: testing, chance, anova, modelselect
