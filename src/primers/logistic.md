# The idea

When the answer is yes or no, a model should give, at each input $x$, the probability of yes. A sloping straight line runs below 0 and above 1. Logistic regression makes the log of the odds a straight line instead, which bends the probability into a smooth step, and fits it by maximum likelihood. The scene then turns probabilities into decisions. You need conditional probability (scene chance) and maximum likelihood (scene likelihood).

# The mathematics

## Odds and the logistic curve

The odds of yes are $p/(1 - p)$: a probability of $0.8$ is odds of 4. The log-odds $\ln\bigl(p/(1 - p)\bigr)$ is 0 at $p = 1/2$ and can be any number, so a straight line can be one. The model, and the same solved for $p$:

$$ \ln\frac{p(x)}{1 - p(x)} = a + bx \qquad\iff\qquad p(x) = \frac{1}{1 + e^{-(a + bx)}} $$

One more unit of $x$ adds $b$ to the log-odds, so it multiplies the odds by $e^b$, the odds ratio. The midpoint, where $p = 1/2$, is at $x = -a/b$; the larger $|b|$, the steeper the step.

Example: $a = 0$ and $b = \ln 3$. At $x = -1, 0, 1, 2$ the odds are $1/3$, 1, 3, 9, and the probabilities $1/4$, $1/2$, $3/4$, $9/10$. Each unit triples the odds; past the midpoint, $p$ moves by less and less.

## Fitting by maximum likelihood

Each data point $x_i$ is a yes or a no. The model gives what happened there the probability $p(x_i)$ if a yes, $1 - p(x_i)$ if a no. Taking the points as independent, the probability of all of it is the product; its log is the log-likelihood:

$$ \ell(a, b) = \sum_{\text{yes}} \ln p(x_i) + \sum_{\text{no}} \ln\bigl(1 - p(x_i)\bigr) $$

The fit takes the $a$ and $b$ that make $\ell$ largest: the top. Under the example's curve, a yes at $x = 1$ and a no at $x = -1$ each get $3/4$, so $\ell = 2 \ln(3/4) = -0.575$. No formula gives the top, but $\ell$ is dome-shaped (concave) in $a$ and $b$, so it has no lower hilltops to stop on, and Newton's method, kept from overshooting, climbs to it.

If every yes lies to the right of every no, there is no top: a steeper curve with its midpoint in the gap gives every point more probability, and $\ell$ creeps up towards 0. A penalty fixes it: with $\lambda > 0$ the fit maximises $\ell - \tfrac{\lambda}{2} b^2$, which falls again once $|b|$ is large.

Why this shape: if the no's are bell-shaped about $-d/2$ and the yes's about $+d/2$, both with spread $\sigma$, the log of the ratio of the two bell curves at $x$ is

$$ \frac{(x + d/2)^2 - (x - d/2)^2}{2\sigma^2} = \frac{d}{\sigma^2}\, x $$

so the true log-odds, this plus the overall log-odds, are a straight line, of slope $d/\sigma^2$.

## Decisions

Say yes wherever $p(x)$ is at least a threshold. The hit rate is the share of the yes's said yes to; the false alarm rate, the share of the no's. Lowering the threshold can never lower either. The ROC curve is hit rate against false alarm rate for every threshold. The area under it is the share of (yes, no) pairs in which the yes gets the higher probability, a tie counting half: $1/2$ for guessing, 1 for perfect. With rare yes's, the share said right can be high for a model that finds none of them.

With two inputs the log-odds are $a + b_1 x_1 + b_2 x_2$, and the boundary $p = 1/2$ is the straight line where that is 0.

# In the picture

- Cyan dots: the yes's, above 1; orange dots: the no's, below 0.
- Yellow: the model $p(x)$ and its handles, the midpoint and the steepness; under it, its log-odds $a + bx$.
- Pink bars: the probability the model did not give to what happened. The shorter, the higher $\ell$.
- Green dashes: the true curve. Blue: the least-squares line. White: the threshold, and its point on the ROC curve.

> Try first "by hand, then by the method": drag the handles to raise $\ell$, then press fit.

see: chance, likelihood, regress, entropy
