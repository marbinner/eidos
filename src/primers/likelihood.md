# The idea

A model says how probable data are for each value of its parameter $\theta$, such as a coin's chance of heads. Maximum likelihood turns this round: keep the data actually seen, and ask which value of $\theta$ makes them most probable. How sharply the likelihood peaks says how sure that estimate is. You need densities, and the standard error from the scene on estimators.

# The mathematics

## The likelihood

Let $f(x \mid \theta)$ be the model's density at $x$ when the parameter is $\theta$ (for a coin's toss, its chance). For $n$ values that come independently, the probability (or density) of them all is the product, and its logarithm a sum:

$$ L(\theta) = \prod_{i=1}^{n} f(x_i \mid \theta), \qquad \ell(\theta) = \ln L(\theta) = \sum_{i=1}^{n} \ln f(x_i \mid \theta) $$

$L(\theta)$ is the likelihood, $\ell(\theta)$ the log-likelihood. The logarithm rises with its argument, so both peak at the same place, the maximum likelihood estimate $\hat\theta$; the sum stays a handy size where the product of many small numbers vanishes. $L$ is a function of $\theta$ with the data fixed, not a distribution over $\theta$: its area need not be 1.

For a coin with $h$ heads in $n$ tosses, $L(p) = p^h (1 - p)^{n - h}$. The slope of $\ell$, $h/p - (n - h)/(1 - p)$, is positive where $p < h/n$ and negative where $p > h/n$, so the peak is at $\hat p = h/n$, the share of heads.

## How sharp the peak is

Near its peak the log-likelihood is often close to a parabola, and the parabola's curvature, the information $-\ell''(\hat\theta)$, gives a standard error:

$$ \text{SE} = \frac{1}{\sqrt{-\ell''(\hat\theta)}} $$

Why: for values from a bell of known spread $\sigma$, $\ell(\mu)$ is exactly a parabola, $\text{constant} - n(\bar x - \mu)^2 / (2\sigma^2)$, with its peak at the mean $\bar x$ and curvature $n/\sigma^2$, and $1/\sqrt{n/\sigma^2} = \sigma/\sqrt{n}$ is the mean's standard error. With much data, and the model right, most smooth log-likelihoods look nearly like this near the peak. The information is a sum of one term per value, so it grows about in proportion to $n$.

## Two intervals

The parabola falls $1.96^2/2 = 1.92$ below the peak at $\hat\theta \pm 1.96\,\text{SE}$. The likelihood interval asks the same of the curve itself: it holds the values whose log-likelihood is within 1.92 of the peak (likelihood at least 0.15 of the peak's). Where the curve is close to its parabola, the two 95 % intervals agree. Where it is lopsided, as with few tosses and a share near 0 or 1, the likelihood interval follows the curve, and the parabola's can run past values that cannot be. If the model is right, in large samples both cover the truth about 95 times in 100.

Example: 7 heads in 10 tosses, so $\ell(p) = 7 \ln p + 3 \ln(1 - p)$. The peak is $\hat p = 0.7$, where $\ell = -6.11$. A fair coin gives $\ell(0.5) = 10 \ln 0.5 = -6.93$, only 0.82 lower, so 0.5 is inside the likelihood interval, which runs from 0.39 to 0.92. The curvature is $7/0.7^2 + 3/0.3^2 = 47.62$, so $\text{SE} = 0.145$ and $\hat p \pm 1.96\,\text{SE}$ runs from 0.42 to 0.98. With 70 heads in 100 the curvature is ten times as large and the SE $\sqrt{10}$ times smaller.

The curve compares values of $\theta$ within one model: a wrong model can give as tidy a peak, with no warning.

# In the picture

- Cyan: the data (for a coin, each toss, filled for heads).
- Purple: the value tried, $\theta$ (drag its mark), the model there, and the data's heights under it and their logarithms.
- White: the log-likelihood $\ell$ (or the likelihood $L$) of every value.
- Yellow: the peak $\hat\theta$, and the likelihood interval on the dashed line 1.92 below it.
- Orange: the parabola with the peak's curvature, and $\hat\theta \pm 1.96\,\text{SE}$.
- Green: the truth the data came from.

> Try first "7 heads in 10": drag the purple mark to the peak, then try "70 heads in 100".

see: distributions, estimators, intervals, bayes
