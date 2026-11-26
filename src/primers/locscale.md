# The idea

A cause can set how widely its effect scatters as well as where it lies: a dose that raises a response on average and also makes it more variable. Then the check of "Cause or effect?" can fail even in the causal direction, as what is left over spreads more where the cause makes it. This scene fits the spread too, divides by it, and checks again; and shows what a hidden common cause does. It builds on "Cause or effect?".

# The mathematics

## Level and spread

The location-scale noise model says

$$ Y = f(X) + g(X)\,N, \quad N \text{ independent of } X $$

where $f$ is the location (the level), $g > 0$ the scale (the spread), and $N$ a noise of mean 0 and spread 1. With $g$ the same everywhere, this is the added noise of "Cause or effect?".

## Divide by the spread

Fit the level alone, and what is left over is $Y - f(X) = g(X)\,N$. Its average is 0 at every $X$, so it shows no trend, but its spread is $g(X)$: unless $g$ is constant, it depends on $X$ although $X$ causes $Y$. Divided by the spread, it is the noise itself, the standardised residual

$$ \frac{Y - \hat f(X)}{\hat g(X)} \approx N $$

independent of $X$ but for the fits' own errors.

Example. Let $X$ be 1 or 3, $f(x) = 2x$, $g(x) = x$ and $N = \pm 1$. At $X = 1$, $Y = 2 \pm 1$, so 1 or 3; at $X = 3$, $Y = 6 \pm 3$, so 3 or 9. Less the level, that leaves $\pm 1$ at $X = 1$ and $\pm 3$ at $X = 3$: a spread that grows. Divided by $g$, it is $\pm 1$ at both.

The scene's usual fit takes two steps: a smooth curve through the points for the level, then a smooth curve through the squares of what is left over, whose square root is the spread. That works because the average of $(g(X)\,N)^2$ at any one $X$ is $g(X)^2$.

## The check and its limits

Each way round, the standardised residual is tested for independence of the input, as in "Cause or effect?" ($p < 0.05$: dependent). One way passing gives that arrow; both passing, "both fit"; neither, "neither fits". In the wrong direction, in general, no level and spread make what is left over independent of the input, so the direction can be told. A theorem, with exceptions in special cases; the familiar one is a straight line, an even spread, and the cause and the noise both bell-shaped. In the scene's counts, a spread fitted where none is needed lets the wrong direction pass more often.

## A hidden common cause

All this assumes that no hidden variable causes both. If a hidden $U$ does, the check can mislead. In the scene, $U$ sets $X$ and the level of $Y$: fitting the level alone gives the wrong arrow $Y \to X$, and fitting the spread too mostly lets both pass. Measure $U$ (then called $Z$) and put it in every fit: both pass, as they should when neither causes the other. A check that passes is not proof that the model is right; one that fails says only that this model does not fit this way round.

# In the picture

- At the top, the truth (green arrows), the buttons that choose the fit, and the verdict.
- On the left $X$ (cyan) is supposed to cause $Y$ (orange), on the right the other way round. Yellow is the fitted level $\hat f$, shaded the band $\hat f \pm \hat g$; green dashed, the truth.
- Under each, the standardised residual against the input, in blue boxes per slice: a level band when independent. The strip compares the real pairing (white) with shuffled ones (grey).
- Purple is the third variable $U$, dimmed while hidden; measured, the dots take its colour.

> Try first "a spread that grows", then press level only: the band on the left becomes a funnel.

see: direction, confound, discover
