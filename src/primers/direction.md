# The idea

Two variables go together, and no experiment settles which drives the other. Their correlation is the same both ways round, so the scatter alone cannot say, unless something more is assumed. This scene assumes that the effect is a function of its cause plus noise unrelated to the cause, and checks that in each direction. It needs curve fitting, correlation and the first half of the course.

# The mathematics

## The additive noise model

If $X$ causes $Y$, the model says

$$ Y = f(X) + N, \quad N \text{ independent of } X $$

where $f$ is a smooth curve and $N$ is noise. Independent is more than uncorrelated (no straight-line trend): in every slice of $X$ the noise has the same spread and shape, not only the same average.

## Fit both ways, look at what is left over

Fit $Y$ from $X$ with a smooth curve $\hat f$. If the model holds, the residuals $Y - \hat f(X)$ are the noise $N$, but for the fit's own small misses, so they are independent of $X$. Fit $X$ from $Y$ with a curve $\hat g$ too. Its residuals $X - \hat g(Y)$ show no trend along $Y$, but in general their spread or shape changes along $Y$. The direction whose residuals are independent of the input is taken as the causal one.

Example. Let $X$ and $N$ each be $-1$ or $1$, all four pairs equally likely, and $Y = X + N$: the points $(-1, -2)$, $(-1, 0)$, $(1, 0)$ and $(1, 2)$.

- From $X$: $\hat f(X) = X$, and what is left over is $N = \pm 1$ at both values of $X$: independent.
- From $Y$: the average $X$ at $Y = -2$, $0$ and $2$ is $-1$, $0$ and $1$, so $\hat g(Y) = Y/2$. What is left over, $X - Y/2$, is $0$ at $Y = \pm 2$ and $\pm 1$ at $Y = 0$. Its average is $0$ at every $Y$, yet it is narrow at both ends and wide in the middle: uncorrelated, but dependent.

## Testing for independence

The scene measures dependence with HSIC, a number that sees dependence of any kind: 0 for independent variables, but for sampling noise. It is judged against shuffles: the residuals dealt out again at random among the inputs, 99 times, which destroys any dependence. If $k$ shuffles look at least as dependent as the real pairing,

$$ p = \frac{k + 1}{100} $$

Below $0.05$ the residuals are called dependent. Independent with $X$ as the input only, the verdict is $X \to Y$; with $Y$ only, $Y \to X$; both ways or neither, it cannot tell.

## When it cannot tell

With a straight line, and the cause and the noise both bell-shaped (normal), both directions fit exactly: the backward residual is bell-shaped too, and for sums of the same independent bell-shaped pieces, uncorrelated means independent. With a bent curve, or a cause or noise that is not bell-shaped, only one direction fits, but for a few special combinations of curve, cause and noise. The method cannot check two things it assumes: that nothing hidden causes both variables, and that the noise is added, not multiplied. With few points, a real dependence can hide in the sampling noise.

# In the picture

- On the left $X$ is supposed to be the cause, on the right $Y$. In each column the supposed cause (cyan) runs across, the supposed effect (orange) up; the fitted curve is yellow.
- Green is the truth (drag its dots to bend the true curve) and right verdicts; pink, wrong ones; purple, a hidden common cause $Z$.
- Below, what is left over against the input; blue boxes show each slice's middle 80 %, middle half and median, all alike when independent. In the strip, grey dots are shuffled pairings, the white line the real one.

> Try first "a bent cause": a level band on the left, a pinched one on the right. Then "a straight line, bell noise".

see: confound, lingam, locscale
