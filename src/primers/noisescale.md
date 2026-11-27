# The idea

A step of stochastic gradient descent (SGD) averages the gradients of $B$ examples, so it costs $B$ examples. A larger batch gives a gradient with less noise, which allows a longer step, but only until the curvature of the loss, not the noise, limits the step; then more examples buy almost nothing. This scene finds the batch size where the limit changes hands, the noise scale, and what it means for the steps and examples of a whole run. You need the gradient, the Hessian, and a mini-batch gradient seen as the true gradient plus noise.

# The mathematics

## One step, on average

Let $G$ be the true gradient (of the loss on all the data), $H$ the Hessian and $\Sigma$ the covariance of one example's gradient. The mean gradient of $B$ independent examples is $G + \xi$, with noise $\xi$ of mean 0 and covariance $\Sigma/B$. Replace the loss by its second-order approximation and step $\eta$ against $G + \xi$. Averaged over the noise, the cross terms vanish and $\mathbb{E}\,\xi^\top H \xi = \operatorname{tr}(H\Sigma)/B$, so

$$ \mathbb{E}\,\Delta L = -\eta\,|G|^2 + \tfrac12\eta^2\Big(G^\top H G + \frac{\operatorname{tr}(H\Sigma)}{B}\Big) $$

This is a parabola in $\eta$. Its lowest point is at $\eta_{\text{opt}} = |G|^2/(G^\top HG + \operatorname{tr}(H\Sigma)/B)$, where the loss falls by $\Delta L_{\text{opt}} = \tfrac12|G|^2\eta_{\text{opt}}$; so the lowest points of all the batches lie on one line, $-\tfrac12|G|^2\eta$. Dividing through by $G^\top HG$:

$$ \eta_{\text{opt}} = \frac{\eta_{\max}}{1 + B_{\text{noise}}/B}, \qquad \Delta L_{\text{opt}} = \frac{\Delta L_{\max}}{1 + B_{\text{noise}}/B}, \qquad B_{\text{noise}} = \frac{\operatorname{tr}(H\Sigma)}{G^\top HG} $$

Here $\eta_{\max} = |G|^2/G^\top HG$ and $\Delta L_{\max} = \tfrac12|G|^4/G^\top HG$ belong to the true gradient. For $B$ far below $B_{\text{noise}}$ the factor $1 + B_{\text{noise}}/B$ is about $B_{\text{noise}}/B$, so the gain of a step grows in proportion to $B$, and the gain per example stays about the same. For $B$ far above, it is about 1: a larger batch hardly helps. At $B = B_{\text{noise}}$ the gain is half the true gradient's. All of this needs the parabola to open upwards, $G^\top HG + \operatorname{tr}(H\Sigma)/B > 0$; where the loss curves down along the gradient, $G^\top HG \le 0$, there is no noise scale.

If $H$ were a multiple of the identity, $B_{\text{noise}}$ would equal the simple noise scale $B_{\text{simple}} = \operatorname{tr}\Sigma/|G|^2$.

Example: the scene's network at its start has $|G|^2 = 0.01803$, $G^\top HG = 0.004872$ and $\operatorname{tr}(H\Sigma) = 2.430$. So $B_{\text{noise}} = 498.8$, $\eta_{\max} = 3.701$ and $\Delta L_{\max} = 0.0334$. With $B = 16$ the factor is $1 + 498.8/16 = 32.2$: the best step is $0.115$ and gains $0.00104$. With $B = 32$ the factor is $16.6$, and the gain $0.00201$, 1.94 times as much.

## A whole run

Suppose every step is the best one and the noise scale stays the same through a run. Then a run at batch $B$ needs $1 + B_{\text{noise}}/B$ times the fewest steps $S_{\min}$, which a very large batch would take, and $1 + B/B_{\text{noise}}$ times the fewest examples $E_{\min}$, which a very small one would use. With $B_{\text{crit}} = E_{\min}/S_{\min}$ (here equal to $B_{\text{noise}}$):

$$ \Big(\frac{S}{S_{\min}} - 1\Big)\Big(\frac{E}{E_{\min}} - 1\Big) = 1 $$

At $B = B_{\text{crit}}$ a run takes twice the fewest steps and twice the fewest examples. In a real run the noise scale changes as the weights move, and the model predicts $B_{\text{crit}}$ from its average over the run; the scene measures how well that works on its own network.

# In the picture

- The parabolas: the expected change of the loss against the step; cyan is the batch chosen, white the true gradient, grey other batches; the yellow tick is $2\eta_{\text{opt}}$, beyond which the loss rises on average.
- The best gain against $B$, on logarithmic axes: a straight rise, a bend at the red $B_{\text{noise}}$, and the white ceiling $\Delta L_{\max}$. Drag the cyan dot.
- Two charts of a whole run: steps against $B$, and steps against examples, from the model with $B_{\text{noise}}$ held fixed, from the noisy quadratic model, or measured on the network.

> Try first "far below the noise scale, and twice it": twice the batch gains nearly twice as much in a step, so a run needs barely more examples.

see: hessian, minibatch, noisefloor, signnoise
