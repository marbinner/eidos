# The idea

Some densities can be worked out at any point, up to a constant factor, yet there is no easy way to draw samples from them directly. A posterior in several unknowns is the usual case: prior times likelihood is easy, but the total that would rescale it runs over every combination of values. Markov chain Monte Carlo (MCMC, Monte Carlo meaning computing by random draws) samples such a density by a walk: propose a move, take it or refuse it by a rule that needs only ratios of the density, and in the long run the places visited spread like the target. You need densities, the law of large numbers and the posterior (Bayesian updating).

# The mathematics

## The target and the walk

The target is a density $p$ on the plane, given by its logarithm up to a constant. The banana is

$$ \log p(x, y) = -x^2/6 - 2\,(y - 0.3x^2 + 1)^2 + \text{constant} $$

The walker stands at $z = (x, y)$. Each step depends only on where it stands, not on how it got there: that makes the walk a Markov chain.

## Metropolis

Propose a nearby point $z'$: each coordinate moves by the step size times a number drawn from the standard bell. Accept it with probability

$$ \min\left(1, \frac{p(z')}{p(z)}\right) $$

and otherwise stay at $z$, which then counts as visited once more. Uphill moves are always taken, downhill ones less often the further down. The unknown constant cancels in the ratio.

Why the places end up spread like $p$: suppose many walkers were already spread so. A move from $z$ to $z'$ is proposed as often as the move back, and is taken with probability $\min(1, p(z')/p(z))$. So the flow from $z$ to $z'$ is in proportion to $\min(p(z), p(z'))$, the same as the flow back. Every pair of places stays in balance, so the spread does not change. A walk that can reach every part of the target settles into this spread from any start, in the long run.

Example: on the banana, the walker at its top, $(0, -1)$, where $\log p = 0$ (taking the constant as 0). The proposal $(1, -1)$ has $\log p = -1/6 - 2 \times 0.3^2 = -0.347$, so it is taken with probability $e^{-0.347} = 0.71$. The move back is uphill and always taken.

Too small a step, and nearly every move is taken but the walker crawls; too large, and nearly every one is refused.

## Using the slope

- Hamiltonian: give the walker a random velocity $v$ and let it roll, like a ball, over the landscape of $-\log p$, in many small leapfrog steps. The end is accepted with probability $\min(1, e^{H_0 - H_1})$, where $H = -\log p + |v|^2/2$ is the energy before and after. Rolling keeps the energy nearly the same, so even long moves are nearly always taken, if the leapfrog steps are small.
- Langevin: lean each random proposal uphill, along the direction in which $\log p$ climbs fastest, with the acceptance rule corrected because a move and its return are no longer proposed equally often.

## What comes out

Each place depends on the last, so places are not independent. Averages over the walk still settle to the target's averages, but usually more slowly than for independent draws. The first places still remember the start, $(-2.5, 2)$: they are the burn-in, which the bars and the mean leave out. With two peaks far apart, a walk may stay on one for a long time.

# In the picture

- The glow: the target, brightest where $p$ is largest.
- White dot: the walker.
- Green: a move it took (for Hamiltonian, the whole path it rolled); red: a proposal it refused.
- Pale dots: where it has been; orange dots: the burn-in.
- Cyan bars along two edges: how often each $x$ and each $y$ was visited; their shapes estimate the target's densities of $x$ alone and of $y$ alone.

> Try first "steps too small", then "steps too large", and compare the share accepted.

see: lln, bayes, gauss2, descent
