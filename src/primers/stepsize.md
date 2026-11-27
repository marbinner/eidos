# The idea

Gradient descent takes steps of a fixed size $\eta$ against the gradient. How long may a step be, and how many does the descent need? On a bowl, a loss that is quadratic, both have exact answers, from one number per direction. You need the Hessian and its eigenvalues (the scenes "quadform" and "hessian").

# The mathematics

## One direction at a time

Put the bottom of the bowl at $0$, and measure the distance $x_i$ from it along each of the bowl's own directions: the eigenvectors of its Hessian, whose eigenvalues $\lambda_i$ are the curvatures. Then

$$ L = \tfrac12 \sum_i \lambda_i x_i^2 $$

and the gradient along direction $i$ is $\lambda_i x_i$, which depends on $x_i$ alone. So a step $w \leftarrow w - \eta \nabla L$ acts on each direction separately:

$$ x_i \leftarrow (1 - \eta\lambda_i)\, x_i $$

After $t$ steps the distance is $(1 - \eta\lambda_i)^t$ times what it was, and that direction's part of the loss $(1 - \eta\lambda_i)^{2t}$ times. The factor says everything:

- between $0$ and $1$: the direction creeps in, never crossing the bottom;
- $0$, at $\eta = 1/\lambda_i$: it is done in one step;
- between $-1$ and $0$: it overshoots, flipping sides at every step, and still shrinks;
- below $-1$, that is $\eta > 2/\lambda_i$: it flips and grows.

## The sharpest direction sets the limit

The descent comes to the bottom from every start exactly when every factor is between $-1$ and $1$. With every curvature above $0$, that is

$$ 0 < \eta < 2/\lambda_{\max} $$

The largest curvature $\lambda_{\max}$, the sharpness, sets the longest safe step. A step a little longer makes that one direction grow while every other still shrinks.

## The flattest direction sets the time

The flattest direction's factor, $1 - \eta\lambda_{\min}$, is near $1$ unless $\eta$ is long; but once the sharpest direction's factor is below $0$, a longer step slows that one. The worst of the two is smallest where they are equal and opposite:

$$ \eta^\ast = \frac{2}{\lambda_{\max} + \lambda_{\min}}, \qquad \max_i |1 - \eta^\ast\lambda_i| = \frac{\kappa - 1}{\kappa + 1} $$

with $\kappa = \lambda_{\max}/\lambda_{\min}$, the condition number. For large $\kappa$ this is about $1 - 2/\kappa$, so shrinking the distance by a factor $e$ takes about $\kappa/2$ steps: the number of steps grows in proportion to $\kappa$.

On a logarithmic scale each direction's part of the loss is a straight line: the steep ones are soon gone, and the long tail is the slowest direction alone.

Example: $\lambda_{\max} = 1$, $\lambda_{\min} = 0.1$, $\kappa = 10$. The limit is $2$ and the best step $2/1.1 = 1.82$, with factors $-0.82$ and $+0.82$. Every direction's loss is multiplied by at most $(9/11)^2 = 0.669$ per step, so from any start the loss is a thousand times lower once $t \ge \ln 1000 / (-\ln 0.669) = 17.2$: within 18 steps. For $\kappa = 100$, within 173.

## A loss that is not a bowl

Near a point, a smooth loss is close to a quadratic whose curvatures are the eigenvalues of the Hessian there, so these rules hold there approximately. One form holds exactly: if the Hessian's largest eigenvalue is at most $\ell$ everywhere, every step shorter than $2/\ell$ lowers the loss (unless the gradient is $0$). A network's curvature, though, changes as it trains: that is for the scenes on the edge of stability.

# In the picture

- The bowl from above: grey level lines; dashed green, its sharpest and flattest directions; cyan, the path from the white start, a dot per step.
- The cyan dot on the board's scale is $\eta$, with marks at $1/\lambda_{\max}$, the best step and the limit $2/\lambda_{\max}$ (yellow).
- Each row is a direction: its curvature $\lambda$ and its factor $1 - \eta\lambda$, a dot on a line from $-1.5$ to $1$; beyond the yellow line at $-1$ it grows.
- The chart: the loss as a share of the start, on a logarithmic scale; cyan the whole loss, faint green each direction's part.

> Try first "the best fixed step": on the chart the two directions' lines run parallel, one direction creeping and the other flipping sides.

see: quadform, hessian, momentum, eostoy
