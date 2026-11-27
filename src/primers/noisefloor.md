# The idea

On a bowl, gradient descent shrinks the distance to the bottom along each of the bowl's directions by a fixed factor every step. Stochastic gradient descent (SGD) steps along a mini-batch's gradient instead: the true gradient plus noise. So each step also adds a random kick, and the kicks never stop. With a constant step the runs settle in a cloud whose expected loss is a floor, set by the step and the batch size. On this bowl it is exact and gives a clear reason to make the step smaller during training. You need gradient descent on a bowl and the idea of a variance.

# The mathematics

## One direction, one step

Along one of the bowl's own directions, with curvature $\lambda$ (an eigenvalue of its Hessian), the loss is $\tfrac12\lambda z^2$, $z$ being the distance from the bottom. A batch of $B$ examples gives the gradient $\lambda z + \xi$, with noise $\xi$ of mean 0. If one example's noise has variance $\sigma^2$ and the examples are independent, their mean has variance $\sigma^2/B$. A step of size $\eta$ gives

$$ z_{t+1} = (1 - \eta\lambda)\,z_t - \eta\,\xi_t $$

Square it and average. The noise is new at each step, so independent of $z_t$, and the cross term averages to zero:

$$ \mathbb{E}\,z_{t+1}^2 = (1 - \eta\lambda)^2\,\mathbb{E}\,z_t^2 + \frac{\eta^2\sigma^2}{B} $$

The mean obeys $\mathbb{E}\,z_{t+1} = (1 - \eta\lambda)\,\mathbb{E}\,z_t$: the cloud's centre moves exactly as noise-free descent would; the noise only spreads it.

## The floor

If $0 < \eta\lambda < 2$ the factor $(1 - \eta\lambda)^2$ is below 1, and $\mathbb{E}\,z^2$ settles where the rule gives back what it is given: $v = (1 - \eta\lambda)^2 v + \eta^2\sigma^2/B$. Solve for $v$ and multiply by $\tfrac12\lambda$:

$$ \text{floor} = \frac{\eta\,\sigma^2}{2B\,(2 - \eta\lambda)} \approx \frac{\eta\,\sigma^2}{4B} $$

The approximation holds when $\eta\lambda$ is small. From a fixed start the whole run is the start's loss $L_0$ forgotten geometrically, plus the floor filled in:

$$ \mathbb{E}\,L_t = (1 - \eta\lambda)^{2t} L_0 + \big(1 - (1 - \eta\lambda)^{2t}\big)\,\text{floor} $$

While the first term is large the runs descend together; once it is small, only the floor is left: the plateau. The bowl's loss is the sum over its directions.

Example: $\eta = 0.1$, $B = 4$, $\sigma^2 = 1$. The direction with $\lambda = 1$ settles at $0.1/(8 \times 1.9) = 0.00658$, the one with $\lambda = 0.25$ at $0.1/(8 \times 1.975) = 0.00633$; together $0.0129$, against $0.0125$ from the approximation.

## Ways down from the floor

- A smaller step lowers the floor (a little more than in proportion), but brings $(1 - \eta\lambda)^2$ nearer to 1: the descent is slower.
- Doubling $B$ halves it exactly, at the same speed per step, for twice the examples.
- Scaling $\eta$ and $B$ together by $k$ keeps $\eta/B$: the floor and the progress per example stay nearly the same while $k\eta\lambda$ is small. As $k\eta$ nears $2/\lambda$ of the sharpest direction, $2 - k\eta\lambda$ goes to 0 and that direction's floor grows without bound.
- Averaging: each run keeps $a_{t+1} = \gamma a_t + (1 - \gamma) z_{t+1}$, a mean over about $1/(1 - \gamma)$ steps. With $r = 1 - \eta\lambda$ its floor is the weights' times

$$ \frac{(1 - \gamma)(1 + r\gamma)}{(1 + \gamma)(1 - r\gamma)} < 1 $$

- Momentum, $v \leftarrow \beta v + g$ and then $\theta \leftarrow \theta - \eta v$, keeps each gradient acting for about $1/(1 - \beta)$ steps, its noise too. For $0 < \eta\lambda < 2 + 2\beta$ the floor becomes

$$ \frac{(1 + \beta)\,\eta\,\sigma^2}{2B\,(2\beta + 2 - \eta\lambda)(1 - \beta)} \approx \frac{\eta\,\sigma^2}{4B\,(1 - \beta)} $$

All of this is exact for a bowl whose noise does not depend on where the weights are; a network's noise does.

# In the picture

- Cyan dots: 400 runs from the white start. The cloud is Gaussian, so the ring of two standard deviations round its predicted centre holds $1 - e^{-2} \approx 86\%$ of them on average.
- On the chart (logarithmic axes) the line is $\mathbb{E}\,L_t$ from the rule, the dots the runs' mean loss, the red dashed line the floor.
- The strip above it shows the step $\eta$ through the run; white is the averaged weights, pink momentum.

> Try first "a constant step", then "half the step": the floor falls to a little under half, and takes longer to reach.

see: stepsize, minibatch, momentum, noisescale
