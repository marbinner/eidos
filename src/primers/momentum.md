# The idea

In a long, narrow valley gradient descent zigzags across the floor and crawls along it: the steep direction limits the step, and at that step the flat direction barely moves. Momentum keeps a share $\beta$ of the previous step, as a heavy ball keeps its speed. This scene shows, on a bowl, which steps and momenta make a direction creep in, ring, flip or fly off, and why, on a bowl, the best pair needs a number of steps that grows like $\sqrt\kappa$ where plain descent's grows like $\kappa$. It helps to know the scene before (step size) and that a quadratic can have complex roots.

# The mathematics

## One direction, two numbers

On a bowl $L = \tfrac12 \sum_i \lambda_i x_i^2$ every direction moves on its own: $x$ is the distance from the bottom along it and $\lambda$ its curvature, so its slope is $g = \lambda x$. Heavy-ball momentum, as PyTorch writes it, keeps a velocity $v$:

$$ v_{t+1} = \beta v_t + \lambda x_t, \qquad x_{t+1} = x_t - \eta\, v_{t+1} $$

Taking $v$ out leaves $x_{t+1} = (1 + \beta - \eta\lambda)\, x_t - \beta\, x_{t-1}$. A guess $x_t = z^t$ works when

$$ z^2 - (1 + \beta - \eta\lambda)\, z + \beta = 0 $$

Its roots $z_1$, $z_2$ are the eigenvalues of the step's $2 \times 2$ matrix, and after many steps the direction shrinks by $|z|$, the larger one's size, per step.

## Four ways to move, and a limit

The roots are equal where $(1 + \beta - \eta\lambda)^2 = 4\beta$, that is at $\eta\lambda = (1 - \sqrt\beta)^2$ and at $(1 + \sqrt\beta)^2$. For $0 < \beta < 1$:

- below the first, two real positive roots: the direction creeps in without crossing the bottom;
- at the first, one double root (critical);
- between the two, a complex pair. Their product is $\beta$, and the product of a pair $z$, $\bar z$ is $|z|^2$, so every such direction shrinks by exactly $\sqrt\beta$ a step, whatever its curvature, overshooting and swinging back (it rings);
- from the second up to $\eta\lambda = 2 + 2\beta$, two negative roots: it changes side every step and shrinks;
- beyond, a root lies below $-1$ (the polynomial at $z = -1$ is $2 + 2\beta - \eta\lambda$) and the direction flies off.

With $\beta = 0.9$ the limit is $\eta\lambda = 3.8$, against 2 without momentum.

## The best pair

Let the curvatures run from $\lambda_{\min}$ to $\lambda_{\max}$ ($\kappa = \lambda_{\max}/\lambda_{\min}$). Put the flattest direction on the band's left edge and the sharpest on its right edge: $\eta\lambda_{\min} = (1 - \sqrt\beta)^2$ and $\eta\lambda_{\max} = (1 + \sqrt\beta)^2$. Dividing, $\sqrt\kappa = (1 + \sqrt\beta)/(1 - \sqrt\beta)$, so every direction shrinks by

$$ \sqrt\beta = \frac{\sqrt\kappa - 1}{\sqrt\kappa + 1} \qquad \text{against} \qquad \frac{\kappa - 1}{\kappa + 1} $$

for plain descent at its best fixed step. For $\kappa = 100$: $9/11 = 0.818$ against $99/101 = 0.980$, so an error shrinks $e$-fold in about 5 steps against 50. This is the rate in the long run, on a bowl whose extreme curvatures are known; the loss can rise for a while first.

## An average of gradients, and Nesterov

Unrolled, $v_{t+1} = g_t + \beta g_{t-1} + \beta^2 g_{t-2} + \cdots$: about the last $1/(1 - \beta)$ gradients count. On a steady slope it builds up to $g/(1 - \beta)$: a direction with $\eta\lambda$ well below $(1 - \beta)^2$ moves as if the step were $1/(1 - \beta)$ times longer. Nesterov's form takes the gradient a step ahead; on a bowl its limit is $\eta\lambda < (2 + 2\beta)/(1 + 2\beta)$, which is 1.357 for $\beta = 0.9$, below plain descent's 2.

# In the picture

- The map: $\eta\lambda$ across (square-root scale), $\beta$ up, coloured by $|z|$; white lines where the roots are equal, the yellow line the limit.
- Pink dots: the bowl's directions at height $\beta$ (drag the large one). Cyan dots: plain descent's, along the bottom. The white ring marks the slowest.
- Below: one direction's distance, the valley from above, and the losses.

> Try first "the same step with momentum 0.9": all five directions land in the band and shrink alike, by $\sqrt{0.9}$ a step.

see: stepsize, hessian, descent, noisefloor
