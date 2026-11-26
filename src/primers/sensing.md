# The idea

A signal $x$ has $N$ entries, only $k$ of them not zero, and nobody knows which. This scene takes $m$ measurements, far fewer than $N$, each a sum of all the entries with random weights, and, with enough of them, can still recover $x$ exactly. The trick is to pick, among the many signals that agree with the measurements, the one whose entries have the least total size. You need the equation $Ax = y$ with fewer equations than unknowns, and its shortest solution.

# The mathematics

## Too few equations

The measurements are $y = Ax$, where $A$ is an $m \times N$ matrix of random numbers, one row per measurement. With $m < N$, infinitely many $x$ satisfy $Ax = y$: add to any one of them a vector that $A$ sends to 0. Two ways to choose:

- the shortest, with the least 2-norm $\sqrt{\sum x_i^2}$. It usually spreads the signal over every entry: a smear.
- the one with the least 1-norm $\sum |x_i|$, the sum of the sizes of the entries. Choosing it is called basis pursuit.

## Why the 1-norm finds sparse answers

The points with 1-norm at most $r$ form a diamond whose corners lie on the axes, where every entry but one is 0. Grow it until it first touches the solutions of $Ax = y$: being pointed, it tends to touch at a corner or an edge, where many entries are 0. There is always a least-1-norm answer with at most $m$ non-zero entries. The round ball of the 2-norm has no corners, and usually touches where no entry is 0.

That this sparse answer is the right one needs enough measurements. Below $m = k$ nothing could work: even knowing where the $k$ entries are, fewer than $k$ equations cannot pin them down. Above it, the chance of exact recovery climbs from almost never to almost always over a narrow range of $m$. For $k$ up to about $N/4$, the scene's trials put the middle of that climb near

$$ m \approx 1.5\, k \ln(N/k) $$

a rule of thumb, not a theorem. For $N = 100$ and $k = 8$ that is $1.5 \times 8 \times \ln 12.5 \approx 30$: far below 100.

## A small example

One measurement of two unknowns: $A$ is the single row $(1, 2)$, the signal is $x = (0, 1)$, so $y = 2$. Every $x$ with $x_1 + 2x_2 = 2$ agrees. The shortest is $(0.4, 0.8)$, with 1-norm $1.2$ and neither entry 0. Along the line, the 1-norm is least at the corner $(0, 1)$, where it is 1: the signal. A signal $(2, 0)$ would give the same $y$, and the same answer $(0, 1)$: one measurement cannot tell them apart.

## How the scene knows it has the least

It looks for a vector $\nu$, one number per measurement, such that every entry of $A^\top \nu$ lies between $-1$ and $1$, and equals the sign of the green answer's entry wherever that is not 0. Then for any $x$ with $Ax = y$,

$$ \sum_i |x_i| \ge \sum_i x_i (A^\top \nu)_i = (Ax)^\top \nu = y^\top \nu $$

The first step holds because no $(A^\top \nu)_i$ is larger than 1 in size. For the green answer it is an equality, so nothing has a smaller 1-norm. In the example, $\nu = \frac{1}{2}$ gives $A^\top \nu = (0.5, 1)$ and $y^\top \nu = 1$.

# In the picture

- The heat map is $A$ (blue below zero, red above) with $y$ beside it; white ticks over $A$ mark the columns the signal's non-zero entries pick.
- The white stems are the signal $x$.
- Green: the least-1-norm answer. Pink: the shortest answer.
- The chart: how often the least 1-norm is exact at each $m$, green for random sums and purple for reading single entries. The yellow handle is the present $m$; the dashed line is $m = k$.

> Try first "enough measurements", then "too few": the same signal, half the measurements, and the green answer turns into a spray of spikes.

see: underdet, highdim, packing, sae
