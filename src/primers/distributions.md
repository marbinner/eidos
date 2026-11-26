# The idea

A random number $X$ is one whose value is left to chance: the number of heads in ten tosses, the wait for a bus. Its distribution says how likely each value is. This scene shows fourteen families of them, each drawn three ways that carry the same information. You need only probability as a share, from the scene on chance.

# The mathematics

## Counts and densities

A count has a probability for each whole number $k$, $P(X = k)$, drawn as a bar at $k$. The bars add up to 1, since $X$ takes exactly one value.

A number that can take any value in a range, such as a wait, has a density $f$ instead: the probability that $X$ lands in an interval is the area under $f$ over it, and the whole area is 1. A single value has probability 0. The density's height is probability per unit of $x$, not a probability, and can exceed 1.

## Three ways to draw one distribution

The cumulative probability is the probability of $x$ or less:

$$ F(x) = P(X \le x) $$

It is the area under $f$ up to $x$ (for a count, the bars up to $x$), rising from 0 to 1. The quantile function reads $F$ the other way round: $Q(p)$ is the $x$ below which a share $p$ lies, the $x$ with $F(x) = p$ (for a count, the least whole number with $F(x) \ge p$). $Q(0.5)$ is the median.

## An example: the binomial

$X$ counts the successes in $n$ independent tries, each a success with probability $p$. One list of $k$ successes and $n - k$ failures has probability $p^k (1-p)^{n-k}$: independent tries' probabilities multiply. There are $\binom{n}{k}$ such lists, the number of ways to choose which $k$ of the $n$ tries succeed:

$$ P(X = k) = \binom{n}{k}\, p^k (1-p)^{n-k} $$

With $n = 10$ and $p = 0.5$, every list has probability $1/2^{10} = 1/1024$. One list has no success, 10 have one, and $\binom{10}{2} = 10 \times 9/2 = 45$ have two. So $F(2) = (1 + 10 + 45)/1024 = 56/1024 \approx 0.055$, and $Q$ reads it back: $Q(56/1024) = 2$.

## Any distribution from a uniform one

Take $U$ uniform from 0 to 1: as likely to fall in any stretch as in another as long. Then $Q(U)$ has the distribution $F$: $Q(U) \le x$ exactly when $U \le F(x)$, which has probability $F(x)$. Evenly spread $U$'s land crowded where $Q$ is flat, sparse where it is steep: that crowding is the density.

## The families

Each family has parameters that pick one member, as $n$ and $p$ pick one binomial, and most have a story:

- Counts: Bernoulli (one try, 1 or 0), binomial, geometric (the try of the first success), Poisson (the events in a stretch of time in which $\lambda$ are expected).
- Waits for random events at rate $\lambda$: exponential (the first), gamma (the $k$-th, for a whole number $k$).
- Between two limits: uniform, and beta (a share between 0 and 1).
- The normal, the bell curve, with mean $\mu$ and standard deviation $\sigma$ (next scene); the standard normal has $\mu = 0$ and $\sigma = 1$, and a number drawn from it is written $Z$. From it come the chi-square (a sum of $k$ independent squared $Z$'s), Student's $t$, the $F$, the log-normal ($e$ to the power of a normal number) and the Cauchy (a ratio of two independent $Z$'s, so heavy-tailed that it has no mean).

# In the picture

- Green: the density (or bars), $F$ and $Q$; the line above marks the mean and one standard deviation either side.
- White: the mark at $x$; drag it. Blue: the probability up to the mark, $F(x)$, three times: an area, a height, and the $p$ with $Q(p) = x$.
- Orange: a second distribution; pink: the greatest gap between the two cumulative curves.
- Cyan: draws, and their histogram.

> Try first "counting successes", and watch the shaded share up to 2 grow as $p$ falls.

see: chance, moments, clt, likelihood
