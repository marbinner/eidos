# The idea

Two training rules, $A$ and $B$, train the same small network from the same starting weights on the same data. Each chart sets the two side by side on one quantity met earlier in the course; this primer gathers their definitions. You need a gradient, the Hessian and its eigenvalues, a mini-batch, and singular values.

# The mathematics

## Steps

The weights $\theta$ are trained to lower the mean loss over all $N$ examples:

$$ L(\theta) = \frac{1}{N} \sum_{n=1}^{N} \ell_n(\theta) $$

A step uses $g$, the mean gradient of a batch of $B$ examples, and a step size $\eta_t$ that a schedule may change: a warm-up raises it in a straight line from 0, a decay lowers it to 0 at the run's end. Plain descent moves by $-\eta_t g$; heavy-ball momentum keeps $v \leftarrow \beta v + g$ and moves by $-\eta_t v$. The steps chart shows $\eta_t$ and $|\Delta\theta|/|\theta|$, how far a step moved the weights against their size.

## Curvature and the level

The sharpness is the largest eigenvalue of the Hessian $H$ of $L$. On a quadratic, plain descent multiplies the distance to the bottom along an eigenvector of curvature $h > 0$ by $1 - \eta h$ at every step, so it shrinks only while $\eta h < 2$: the level is $2/\eta$. With heavy-ball momentum, $0 \le \beta < 1$, a direction shrinks only while

$$ \eta h < 2 + 2\beta $$

A rule that divides each weight's step by a number $P$ meets the curvature of $P^{-1}H$, the preconditioned sharpness. With $P$ held fixed, Adam is heavy ball written as an average, $m \leftarrow \beta_1 m + (1 - \beta_1) g$, whose level on $P^{-1}H$ is $(2 + 2\beta_1)/((1 - \beta_1)\eta)$. A rule that takes signs or orthogonalises is not a fixed multiple of the gradient and has no such level.

Example: at $\eta = 0.1$ and $\beta = 0.9$ momentum's level is $3.8/0.1 = 38$, plain descent's $2/0.1 = 20$.

## Noise

A batch's gradient has mean $G$, the gradient of $L$. If its examples are drawn independently, its covariance is $\Sigma/B$, with $\Sigma$ that of one example's gradient, and

$$ \mathbb{E}\,|g|^2 = |G|^2 + \frac{\operatorname{tr}\Sigma}{B} $$

The simple noise scale $B_\text{simple} = \operatorname{tr}\Sigma/|G|^2$ is the batch at which the two parts are equal. Example: at $B_\text{simple} = 167$ a batch of 8 has $\mathbb{E}\,|g|^2 = |G|^2 (1 + 167/8) \approx 21.9\,|G|^2$, 95 % of it noise. Here examples are drawn without putting them back, which makes the noise smaller by $(N - B)/(N - 1)$.

## Updates and weights

A matrix's change has singular values $\sigma_1 \ge \sigma_2 \ge \dots$, its stretch along each of its own directions, and stable rank

$$ \frac{\sigma_1^2 + \sigma_2^2 + \cdots}{\sigma_1^2} $$

about how many directions it stretches nearly as much as the most. Muon steps along nearly $U V^\top$ for its momentum $U S V^\top$, whose singular values are all near 1 where the momentum's are not 0: no direction dominates. For an MLP's matrix, plain descent's change without weight decay is a sum of $B$ outer products, one per example, so its rank is at most $B$.

A matrix's size is its Frobenius norm $\|W\|_F$, the root of the sum of its squared entries (and of its squared singular values). Weight decay taken off the weights moves them by $-\eta\lambda\theta$; added to the gradient, it gives the rule $g + \lambda\theta$. For plain descent the two are the same steps; Adam divides the added part by its running size too.

# In the picture

- Orchid is $A$ and lime is $B$, in every chart.
- Sharpness: dots measured every 10 steps (preconditioned for AdaGrad, RMSProp and Adam), and each rule's level $c/\eta_t$ in yellow, dashed.
- Noise: $B_\text{simple}$ with each design's batch size, dashed.
- Update: the singular values of one matrix's last change. Weights: $\|W\|_F$ solid and $\sigma_1$ dashed, for every matrix.

> Try first "plain descent against momentum": the same step, one rule much faster, and each rule's sharpness under its own level.

see: stepsize, momentum, noisescale, muon
