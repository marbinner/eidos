# The idea

The step size of a training run often climbs from almost nothing over the first steps (a warm-up) and falls to almost nothing over the last (a decay). This scene follows one reason for each end, measured on one small network: at the start a sharply curved loss makes a long step unstable; at the end the noise of a mini-batch holds a constant step's loss up. You need gradient descent, the Hessian's eigenvalues and mini-batches.

# The mathematics

## The ceiling the curvature sets

On a quadratic, gradient descent with step $\eta$ multiplies the error along each eigenvector of the Hessian, whose eigenvalue is $h$, by the same factor at every step:

$$ x_{t+1} = (1 - \eta h)\, x_t $$

The error shrinks only if $|1 - \eta h| < 1$, that is $0 < \eta h < 2$. If every eigenvalue is positive, the largest, the sharpness $S$, decides for all of them:

$$ \eta < \frac{2}{S} $$

Read one way it is a ceiling on the step, $2/S$; the other way, a level on the sharpness, $2/\eta$. Over it, the error along the sharpest direction flips sign each step and grows.

Example: at the starting weights $S_0 = 88.3$, a ceiling of $2/88.3 = 0.0227$. With $\eta = 0.09$ the sharpest direction's factor is $1 - 0.09 \times 88.3 \approx -6.9$: on a quadratic its error would grow almost sevenfold per step.

A network is not a quadratic: its sharpness moves with the weights. Here a step a few times over the ceiling made the loss jump while the sharpness fell under $2/\eta$, and training went on (a catapult); a longer step ran away. A warm-up, starting under the ceiling, gave the sharpness time to come down.

## Adam

Adam divides each parameter's step by its own number $P_i$, from a running average of its squared gradients. With $P$ held fixed, Adam is gradient descent with momentum on a stretched loss whose curvatures are the eigenvalues of $P^{-1}H$; the largest is the preconditioned sharpness $\tilde S$. With Adam's averaged momentum, $\beta_1 = 0.9$, the limit on a quadratic is

$$ \tilde S < \frac{2 + 2\beta_1}{(1 - \beta_1)\,\eta} = \frac{38}{\eta} $$

At the first step $P_i \approx (1 - \beta_1)|g_i|$, a tenth of the gradient's size, so $\tilde S$ starts high.

## The ceiling the noise sets

A mini-batch of $B$ independent examples gives the whole data's gradient $G$ plus noise; $\Sigma$ is the covariance of single examples' gradients. In a model of one step of SGD, with the loss expanded to second order, a step $\varepsilon$ changes it on average by

$$ \Delta L = -\varepsilon\, |G|^2 + \tfrac12\, \varepsilon^2 \left( G^\top H G + \operatorname{tr}(H\Sigma)/B \right) $$

The first term is the gain; the second, the cost of the curvature along the gradient and of the noise. So, if the bracket is positive, $\Delta L < 0$ only for

$$ \varepsilon < \frac{2\,|G|^2}{G^\top H G + \operatorname{tr}(H\Sigma)/B} $$

This is the model's noise ceiling, a guide rather than a law. Here, late in training, the noise term sets most of the denominator, the ceiling wanders round a constant step, and the loss wanders on a floor (scene noisefloor). A shorter step goes lower; so does a larger batch.

## The schedules

The warm-up climbs in a straight line, $\eta_t = \eta_{\text{peak}}\, t / T_w$, over the first $T_w$ steps. After it a linear decay falls straight to 0 by the last step; a cosine decay $T$ steps long is, $s$ steps in,

$$ \eta = \tfrac12\, \eta_{\text{peak}} \left( 1 + \cos(\pi s / T) \right) $$

flat at first, steepest halfway, flat near 0. A third of the way from a peak of 0.1: $\tfrac12 \times 0.1 \times (1 + \cos 60^\circ) = 0.075$.

# In the picture

- Top: each run's step $\eta$ in its rule's colour (one run bright, one pale), its ceiling $2/S$ ($38/\tilde S$ for Adam) in yellow, and with mini-batches the noise ceiling as red dots.
- Middle: the sharpness $S$ (or $\tilde S$) in green, the level $2/\eta$ (or $38/\eta$) in yellow.
- Bottom: the whole data's loss.

> Try first "the same peak, warmed up": watch the yellow ceiling rise as the step climbs to meet it.

see: stepsize, eos, noisefloor, adam
