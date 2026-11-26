# The idea

A loss is a number to be made as small as possible by changing a model's parameters; here there are two, $x$ and $y$, so the loss is a landscape. An optimiser is a rule that turns the slope where it stands into a step. You need the gradient: the arrow of slopes that points straight uphill.

# The mathematics

## Gradient descent

Gradient descent steps downhill, further where the slope is steeper:

$$ p_{\text{new}} = p - \eta\, \nabla L(p) $$

Here $p = (x, y)$ is where the optimiser stands, $L$ the loss, $\nabla L$ its gradient and $\eta$ the learning rate. Near the bottom of a smooth bowl the slope flattens, so the steps shrink on their own.

How large can $\eta$ be? That depends on the curvature: how fast the slope changes. Along a direction where $L = \frac{c}{2} y^2$, the slope is $cy$ and the curvature $c$; one step gives

$$ y_{\text{new}} = y - \eta c y = (1 - \eta c)\, y $$

With $\eta c$ below 1 the step falls short of the bottom; between 1 and 2 it jumps across and lands closer; above 2 it lands further out each time. So $\eta$ must stay below $2/c$ for the most sharply curved direction, which makes it slow along gently curved ones.

Example: the ravine $L = 0.05x^2 + y^2$ with $\eta = 0.9$, from $(-2.6, 1.6)$. The gradient is $(0.1x, 2y) = (-0.26, 3.2)$, so the first step lands at $(-2.366, -1.28)$. Across the valley the curvature is 2, and $y$ is multiplied by $1 - 0.9 \times 2 = -0.8$ every step: it bounces from wall to wall. Along it the curvature is $0.1$, and $x$ is multiplied by $1 - 0.9 \times 0.1 = 0.91$: a crawl. A larger $\eta$ would speed the crawl, but must stay below $2/2 = 1$.

## Momentum

Momentum keeps part of the previous step, as a rolling ball keeps its speed:

$$ v_{\text{new}} = \mu v - \eta\, \nabla L(p), \qquad p_{\text{new}} = p + v_{\text{new}} $$

where $v$ is the velocity and $\mu$, the "momentum" setting, is between 0 and 1. On a long, even slope $v$ settles where $v = \mu v - \eta \nabla L$, that is at $-\eta \nabla L/(1 - \mu)$: steps $1/(1 - \mu)$ times as long (ten times for $\mu = 0.9$). Pushes that keep changing sign largely cancel. So in a ravine, with $\mu$ not too large, momentum speeds the crawl and calms the bouncing; the price is that it can overshoot a minimum and swing back.

## Adaptive step sizes

AdaGrad, RMSProp and Adam divide each coordinate of the gradient by that coordinate's typical size: the root of an average of its recent squares (for AdaGrad, of the sum of all past ones). Once the slope has been steady for a while, RMSProp and Adam step about the adaptive rate in each coordinate, however steep or shallow it is: the steep direction is slowed, the shallow one sped up. But this works axis by axis, so it straightens only valleys lined up with $x$ and $y$. And the steps no longer shrink in proportion to the slope, so RMSProp and Adam can take a while to come to rest. Adam adds momentum as well.

Newton's method, which also measures the curvature, heads for any place where the slope is 0: a minimum, but as readily a saddle or a hilltop.

# In the picture

- The surface's height is the loss; the map below is the same seen from above.
- Each coloured ball and trail is one optimiser, coloured as its name in the panel; the white ball is the start.
- Dots on the map: one per step, so their spacing shows the step lengths. White arrows point downhill from each optimiser, along $-\nabla L$.
- The chart: each optimiser's loss, above the lowest point on the map, against the step, on a log scale.

> Try first "learning rate 0.5" in the opening ravine: now $\eta c = 1$ across, so one step lands on the floor, but $x$ is multiplied by $0.95$ a step, not $0.91$.

see: surface, quadform, race, landscape
