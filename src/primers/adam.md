# The idea

Training moves each weight a little against its gradient, the slope of the loss along that weight. Gradient descent moves it by the learning rate $\eta$ times the gradient, so a step is as large as the gradient, and any noise in the gradient goes into the step in full. Adam keeps two running averages for every weight and divides one by the root of the other. The step that comes out is set by $\eta$ and by how steady the gradient has been, not by how large it is. Only averages are needed.

# The mathematics

## Two running averages

Write $g_t$ for one weight's gradient at step $t$. Adam keeps

$$ m_t = \beta_1 m_{t-1} + (1-\beta_1)\, g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2 $$

from $m_0 = v_0 = 0$. At every step the share of each earlier gradient is multiplied by $\beta_1$, so $m$ remembers about the last $1/(1-\beta_1)$ gradients and $v$ the last $1/(1-\beta_2)$ squares: about 10 and 1000 with the defaults, $\beta_1 = 0.9$ and $\beta_2 = 0.999$.

## The corrections for the start at zero

Because the sums start at zero, the weights in $m_t$ add up to $1 - \beta_1^t$, not 1. Dividing by that sum gives a true weighted average:

$$ \hat m_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^t} $$

For a gradient that never changes, $\hat m_t = g$ and $\hat v_t = g^2$ exactly, from the first step on.

## The step

$$ \Delta w_t = -\eta\, \frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon} $$

Here $\varepsilon$, $10^{-8}$ by default, keeps the divisor from being 0. Multiply every gradient by a number $c$: $\hat m$ is multiplied by $c$ and $\sqrt{\hat v}$ by $|c|$, so the step does not change while $\sqrt{\hat v}$ stays far above $\varepsilon$. Far below it, the divisor is about $\varepsilon$ alone and the step is $\eta \hat m/\varepsilon$: gradient descent with momentum.

Example, with $g_1 = 2$: $m_1 = 0.1 \times 2 = 0.2$ and $v_1 = 0.001 \times 4 = 0.004$. Corrected, $\hat m_1 = 0.2/0.1 = 2$ and $\hat v_1 = 0.004/0.001 = 4$, so the first step is $\eta \times 2/\sqrt 4 = \eta$, whatever $g_1$ was. Without the corrections it is $0.2/\sqrt{0.004} = 3.16$ times $\eta$.

## Signal and noise

Let the gradient be a steady part $\bar g$ plus noise of spread $\sigma$, both changing slowly next to the memories, and leave $\varepsilon$ out. Then $\hat m \approx \bar g$ and $\hat v \approx \bar g^2 + \sigma^2$, so on average

$$ \frac{\hat m}{\sqrt{\hat v}} \approx \frac{\bar g}{\sqrt{\bar g^2 + \sigma^2}} $$

which lies between $-1$ and 1. With little noise it is close to $\pm 1$: the weight moves by about $\eta$ in the direction of its gradient's sign, which is sign descent. With much noise it is close to $\bar g/\sigma$, which is small: Adam brakes a weight whose gradients are mostly noise. For $\bar g = 0.1$ and $\sigma = 1$ it is $0.1/\sqrt{1.01} = 0.0995$.

## One divisor for every weight

Each weight has its own $v$, so each axis is rescaled on its own. That suits a valley steep along one axis and shallow along the other, and not the same valley turned between them: in the scene Adam needs 101 steps for the first and 379 for the second. It also changes weight decay. Added to the gradient as $\lambda w$, the decay is divided by $\sqrt{\hat v}$ with the rest, so, while $\lambda w$ is small next to a weight's gradients, a weight with large gradients is hardly decayed. AdamW instead takes $\eta \lambda w$ off every weight directly.

# In the picture

- Upper chart: the red dots are the gradients $g_t$, the white dashes their steady part, the orange line $\hat m_t$ and the light orange lines $\pm(\sqrt{\hat v_t} + \varepsilon)$.
- Lower chart: the step $\hat m_t/(\sqrt{\hat v_t} + \varepsilon)$ in units of $\eta$, with dashed lines at $\pm 1$.
- The squares: two weights from above, with the paths of gradient descent (cyan) and Adam (orange) in a valley, or under the two kinds of weight decay.

> Try first "a steady gradient", where every step is about $\eta$; then "mostly noise", where Adam brakes.

see: descent, steepest, signnoise, wdecay
