# The idea

A curve fitted by least squares fits its own points at least as well when given more freedom, so the error on them cannot say how much freedom is right. What matters is the error on new data, and cross-validation estimates it from the data at hand: hold some points back, fit the rest, and measure the misses on the ones held back. Here the truth is known, so the true error is drawn too, to judge it by. You need least-squares fitting (scene regress) and the idea that an estimate has its own error (scene estimators).

# The mathematics

## Two errors

There are $n$ points $(x_i, y_i)$, each $y_i$ the true curve $f(x_i)$ plus noise of mean 0 and variance $\sigma^2$. Let $\hat y(x)$ be the curve fitted to them.

The training error is the mean squared miss on the points fitted to. It can only fall or stay level as the degree grows, since a higher degree can do all that a lower one can.

The true error is the mean squared miss on a new point. That misses by $f(x) - \hat y(x)$ plus its own noise, which is independent of the fit and averages 0, so the squares add:

$$ \text{true error} = \sigma^2 + \text{the average of } \bigl(f(x) - \hat y(x)\bigr)^2 $$

averaged over $x$ from $-1$ to $1$: never below $\sigma^2$. Training error is too hopeful on average, as the curve has partly followed the noise of the very points it is judged on.

## Cross-validation

Deal the points into $k$ folds. Hold back each fold in turn, fit the curve to the other points, and write $\hat y^{(-i)}$ for the fit that did not see point $i$. Then

$$ \text{CV} = \frac{1}{n} \sum_{i=1}^{n} \bigl(y_i - \hat y^{(-i)}(x_i)\bigr)^2 $$

Each miss is honest: no point's own noise was in its fit. The degree with the lowest CV is chosen. Leave one out is $k = n$: every fit uses $n - 1$ points, nearly all, so CV is then nearly unbiased; with 5 folds each fit has four fifths of the points, and CV tends to come out high.

Example, leaving one out: three points, $(0, 0)$, $(1, 3)$, $(2, 0)$. A flat line (degree 0) fitted to all three sits at their mean, 1, with training error $(1 + 4 + 1)/3 = 2$. Left out in turn, each point is predicted by the mean of the other two, $1.5$, $0$, $1.5$: misses $1.5$, $3$, $1.5$, so CV $= (2.25 + 9 + 2.25)/3 = 4.5$. The best straight line (degree 1) for all three is that same flat line, with training error 2. But a line through any two of the points predicts the third at 6, 0 and 6: misses 6, 3, 6, and CV $= (36 + 9 + 36)/3 = 27$. Training error cannot tell them apart; CV can.

## Choosing has a price

- The orange band is CV plus or minus one standard error: the spread of the folds' own means divided by $\sqrt k$, a rough measure, since the folds share points in their fits.
- A ridge penalty $\lambda$ on the size of the coefficients is another dial from wild to flat, chosen the same way.
- More points pin down more coefficients, so the best degree tends to grow with them.
- The lowest of many held-back errors is partly the luckiest, so the winner's is too low on average: a final judgement needs a test set that played no part in any fit or choice.

# In the picture

- Left: the points fitted to (cyan), those held back (orange) with their misses (pink), the fitted curve (yellow) and the truth (green).
- The error chart, by degree: training error (cyan), CV (orange, its lowest point ringed), the true error (green), and $\sigma^2$ (green dashes). The white dot marks the degree shown on the left; drag it.
- Below: the degree CV chose on 200 new data sets.

> Try first "five folds": watch each fold held back in turn, then press "deal again".

see: regress, estimators, pvalues, multireg
