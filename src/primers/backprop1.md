# The idea

Training a network needs the loss's slope with respect to every weight. Backpropagation finds them all in one sweep back from the loss, using the chain rule at every step. This scene does that sweep with every number shown, for a network of two inputs, two hidden units with $\tanh$, one output and nine weights. You need the chain rule and what a network is.

# The mathematics

## Forward

Each hidden unit $j$ weighs its inputs, adds a bias and passes the sum through $\tanh$; the output weighs the two units the same way; the loss measures how far the output $\hat y$ is from the target $y$:

$$ z_j = w_{j1} x_1 + w_{j2} x_2 + b_j, \quad a_j = \tanh z_j, \\ \hat y = v_1 a_1 + v_2 a_2 + c, \quad L = \tfrac12 (\hat y - y)^2 $$

## Back, a layer at a time

The slope of $L$ with respect to the output is $\delta = \hat y - y$. The output is $v_1 a_1 + v_2 a_2 + c$, so

$$ \partial L/\partial v_j = a_j \delta, \quad \partial L/\partial c = \delta, \quad \partial L/\partial a_j = v_j \delta $$

The slope of $\tanh$ is $1 - \tanh^2$, so going back through unit $j$ multiplies by $1 - a_j^2$:

$$ \delta_j = \partial L/\partial z_j = (1 - a_j^2)\, \partial L/\partial a_j, \quad \partial L/\partial w_{jk} = x_k \delta_j, \quad \partial L/\partial b_j = \delta_j $$

Each line is the chain rule. The loss depends on $w_{12}$ only through $z_1$, and $z_1$ changes by $x_2$ for each unit of $w_{12}$, so $\partial L/\partial w_{12} = x_2 \delta_1$. The pattern is the same on every wire: a weight's slope is what came in on its wire going forward, times what came back to the unit it feeds. A bias is a weight on a wire that always carries 1.

With the scene's starting numbers $x = (1, 0.5)$, $y = 1$: $z_1 = 0.5 - 0.15 + 0.1 = 0.45$ and $a_1 = \tanh 0.45 \approx 0.422$; $z_2 = -0.2$ and $a_2 \approx -0.197$; $\hat y \approx 0.514$ and $L \approx 0.118$. Back: $\delta \approx -0.486$, $\partial L/\partial a_1 = 0.7 \times (-0.486) \approx -0.340$, $1 - a_1^2 \approx 0.822$, so $\delta_1 \approx -0.280$ and $\partial L/\partial w_{12} = 0.5 \times (-0.280) = -0.140$.

## Checking a slope, and where it vanishes

A slope foretells what a small nudge $h$ of one weight does: the loss changes by about $(\partial L/\partial w)\, h$. For $w_{12}$ raised by 0.1 the slope foretells a change of about $-0.140 \times 0.1 = -0.0140$; the loss changes by $-0.0133$. The gap, from the curve bending away from its tangent, shrinks faster than $h$ does.

Where $z_j$ is far from 0, $\tanh$ is flat and $1 - a_j^2$ is nearly 0. Then $\delta_j$ is that small share of what reaches $a_j$, and so are the slopes of every weight feeding unit $j$: the unit is saturated and hardly learns, however wrong it is. At $z_1 = -4$ the factor is about 0.00134.

## Downhill

A step of gradient descent moves every weight against its slope, with a rate $\eta$:

$$ w \leftarrow w - \eta\, \partial L/\partial w $$

For a small $\eta$ the loss falls by about $\eta$ times the sum of the squared slopes; too long a step overshoots and lands higher than it started. With four examples, $L$ is the mean of their own losses $\ell_i = \tfrac12 (\hat y - y)^2$, and every slope is the mean of their four slopes. Two hidden units that start exactly alike (weights in, bias, weight out) get the same slopes, so they stay alike and act as one unit.

# In the picture

- Cyan numbers go forward: the inputs, each unit's $z$ and $a$, the output $\hat y$. Pink numbers come back: $\delta$, $\partial L/\partial a_j$, $\delta_j$, and under each weight its slope.
- Yellow numbers on the wires are the weights; a wire is as wide as its weight is large, and dashed where it is negative. The circles marked 1 feed the biases.
- Orange: the target $y$ and the loss $L$.
- Inside each unit, a little graph of $\tanh$ with the unit's $z$ on it and the slope $1 - a_j^2$ there.
- Under the network, the arithmetic of the step shown; below it, the output over the plane of inputs and charts of the loss.

> Try first "forwards, then back": a step at a time, each with its arithmetic.

see: chain, autodiff, mlp, backpropmat
