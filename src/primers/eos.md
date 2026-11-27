# The idea

On a bowl, gradient descent with a step $\eta$ blows up along any direction whose curvature is over $2/\eta$. A network's loss is not a bowl, and its largest curvature, the sharpness $S$, changes as it trains. This scene trains a small network, measures $S$ at every step, and sets it against the largest curvature the step can bear. You need the Hessian and its eigenvalues, and gradient descent on a bowl.

# The mathematics

## The largest curvature a step can bear

On a bowl the loss is $\tfrac12 \theta^\top H \theta$. Along an eigenvector of $H$ with eigenvalue $\lambda$, let $x$ be the distance from the bottom. One step of gradient descent multiplies it by a fixed factor:

$$ x_{t+1} = (1 - \eta\lambda)\, x_t $$

It shrinks if $0 < \eta\lambda < 2$. Over that the factor is below $-1$: the point flips to the other side at every step, further out each time. So the sharpness, the largest eigenvalue of $H$, must stay under the level $2/\eta$.

Example: $S = 12.5$ and $\eta = 0.2$ give $1 - \eta S = -1.5$. Each step flips the point across and takes it 1.5 times as far out: after ten steps, $1.5^{10} \approx 57.7$ times as far.

Other rules have other levels. Heavy-ball momentum, $v \leftarrow \beta v + g$ and then $\theta \leftarrow \theta - \eta v$, gives $x_{t+1} = (1 + \beta - \eta\lambda)\, x_t - \beta x_{t-1}$, which shrinks exactly when

$$ \eta\lambda < 2 + 2\beta $$

for $0 \le \beta < 1$: with $\beta = 0.9$, curvatures up to $3.8/\eta$. Adam divides each parameter's step by a number of its own, $p_i$. With those held fixed, it is momentum (an average, $m \leftarrow \beta_1 m + (1 - \beta_1) g$) on the loss in the weights $\sqrt{p_i}\,\theta_i$, whose Hessian has the eigenvalues of $P^{-1}H$ ($P$ the diagonal matrix of the $p_i$). The largest is the preconditioned sharpness, with the level

$$ \frac{2 + 2\beta_1}{(1 - \beta_1)\,\eta} = \frac{38}{\eta} \quad \text{for } \beta_1 = 0.9 $$

## Measuring the sharpness

Here $H$ is 97 by 97 and is never written out. Its product with a direction $v$ costs two gradients: $Hv \approx (\nabla L(\theta + rv) - \nabla L(\theta - rv))/2r$ for a small $r$. Lanczos's method builds from a few such products the small matrix of $H$ in a few directions. Its largest eigenvalue $\mu$, with its direction $y$, estimates $S$, and the residual $|Hy - \mu y|$ says how well: some eigenvalue of $H$ lies within it of $\mu$.

## Two phases, and the slice

Cohen and co-authors (2021) trained many networks by full-batch gradient descent and found, by experiment, two phases. First the sharpness rises while it is under $2/\eta$. Then it reaches $2/\eta$ and stays right at, or just above, it: the edge of stability, where the loss goes up and down from step to step but falls over long stretches. For full-batch Adam they report the same of the preconditioned sharpness, at $38/\eta$ (2022). It is observed, not proved.

The slice shows what too sharp for the step looks like. Through the middle $c$ of the last two steps, along the sharpest direction $u$ (a unit eigenvector), the loss is

$$ \ell(s) = L(c + s\,u), \qquad \ell''(0) = u^\top H u \approx S $$

($u$ and $S$ are those of the last iterate, half a step from $c$). Near its middle it is a parabola of curvature $S$. One narrower than a parabola of curvature $2/\eta$ is too sharp for the step, and the iterates flip from wall to wall across it.

# In the picture

- The first chart: the loss at every step, on a logarithmic scale.
- The second: in green, the sharpness $S$ at every step; yellow dashed, the level $2/\eta$ (or $3.8/\eta$, or $38/\eta$ for the preconditioned sharpness).
- The slice: white is $\ell(s)$, green the parabola of curvature $S$, yellow dashed one of curvature $2/\eta$ with the same bottom; the dots are the last iterates, placed by their distance along $u$.

> Try first "a longer step": watch the green line reach the yellow one, and the dots begin to flip.

see: stepsize, hessian, eostoy, warmup
