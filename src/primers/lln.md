# The idea

Roll a die again and again, and the average of the rolls so far settles on 3.5. That is the law of large numbers, the reason an average of many measurements can be trusted. The scene also shows how fast the average settles, and a distribution for which it never does. It uses the mean $\mu$ and standard deviation $\sigma$ from "Mean, spread and shape".

# The mathematics

## The law

Independent draws $x_1, x_2, \dots$ are each made the same way, none affected by the others: a die does not remember its last roll. The average of the first $n$ is the sample mean,

$$ \bar x_n = \frac{x_1 + x_2 + \dots + x_n}{n} $$

The law of large numbers: if the distribution has a mean $\mu$, then, with probability 1, $\bar x_n$ comes within any margin of $\mu$, however small, and from some $n$ on stays within it. It does not say how soon.

## How fast

Each draw misses $\mu$ by $d_i = x_i - \mu$, and the total misses $n\mu$ by $d_1 + \dots + d_n$, the gap in the total. Square that sum: it is the squares $d_i^2$ plus the cross products $d_i d_j$. For independent draws each cross product averages to 0, since $d_j$ averages 0 whatever $d_i$ is. So the gap's mean square is $n\sigma^2$, the variances add up, and its typical size is $\sigma\sqrt{n}$: it grows. The average's distance from $\mu$ is the gap divided by $n$, so its typical size, the standard error, shrinks:

$$ \frac{\sigma\sqrt{n}}{n} = \frac{\sigma}{\sqrt{n}} $$

(Typical means the square root of the mean square, over many runs; this needs $\sigma$ to be finite.) Four times the draws halve it; a hundred times, a tenth.

Example: a die has $\mu = (1 + 2 + \dots + 6)/6 = 3.5$. The squared distances of the faces from 3.5 are 6.25, 2.25, 0.25, 0.25, 2.25 and 6.25, whose mean is $17.5/6 \approx 2.92$, so $\sigma \approx 1.71$. After 10 rolls the typical distance is $1.71/\sqrt{10} \approx 0.54$; after 1 000 it is about 0.054.

## The funnel

The funnel is

$$ \mu \pm 1.96\, \frac{\sigma}{\sqrt{n}} $$

At each $n$ about 95 % of runs are inside it once the averages are spread like a bell, which the central limit theorem (the next scene) says happens as $n$ grows. At every $n$ a run is inside with probability at least $1 - 1/1.96^2 \approx 0.74$, by Chebyshev's inequality. It speaks of each $n$ alone: a run inside at one $n$ can be outside at another.

## No memory, no mean, and $\pi$

A fair coin after ten tails still gives heads half the time. The share of heads returns to one half because the ten tails become a smaller and smaller part of the tosses, not because heads catch up: the count of heads less half the tosses stays 5 short on average, and its typical distance from 0 grows.

The Cauchy distribution has no mean, and an average of any number of its draws is spread exactly as one draw is: it never settles.

A point thrown at random into a square lands in the circle that fits inside it with probability $\pi/4$, the circle's share of the area. Score 4 for a point inside and 0 for one outside: the mean score is $\pi$, and the average closes in on it. Working a number out as an average of random draws is the Monte Carlo method.

# In the picture

- Yellow: $\bar x_n$ against $n$, one path for each run; bold, the run being drawn now.
- Green: $\mu$ and the funnel round it; on the right, the chances or density of one draw.
- Cyan: the bold run's draws so far.
- Orange, below: the gap in the total, against a band that widens like $\sqrt{n}$.
- White: two markers on $n$ to drag, where the table measures the runs; pink: a run outside the funnel there.

> Try first "a die", and compare the typical distance at the two markers with $\sigma/\sqrt{n}$.

see: moments, clt, intervals, estimators
