# The idea

Backpropagation hands the loss's gradient back from the output, layer by layer. In a plain deep network (no normalisation, no skip connections) every layer multiplies what comes back by its weights and by its gates. Fifty layers are fifty such factors: unless each keeps the size on average, what reaches the first layers shrinks or grows geometrically. The scene measures this, beside a prediction. You need backpropagation through one layer, and gates.

# The mathematics

## One layer, forward and back

Hidden layer $l$ has $n$ units, each reading the $n$ outputs of the layer before. It computes $z_l = W_l h_{l-1} + b_l$ and its output $h_l = \phi(z_l)$. Starting weights are independent, of mean 0 and variance $\sigma^2/n$; biases start at 0. Each entry of $z_l$ is a sum of $n$ random terms, so in a wide layer it is spread like a bell whose mean square is $\sigma^2$ times that of $h_{l-1}$.

Going back, write $g_l$ for $\partial L/\partial h_l$. The chain rule gives

$$ g_{l-1} = W_l^\top \big(\phi'(z_l) \odot g_l\big) $$

where $\odot$ multiplies entry by entry: each unit's gradient is multiplied by its gate $\phi'(z_l)$, then spread back through the transposed weights. Each entry of $g_{l-1}$ is again a sum of $n$ random terms; for wide layers its mean square is

$$ \overline{g_{l-1}^2} \approx \chi \, \overline{g_l^2}, \qquad \chi = \sigma^2 \, \mathbb{E}\big[\phi'(z)^2\big] $$

Where $\chi$ is the same at every layer, the root mean square changes by $\sqrt{\chi}$ a layer: a straight line on a logarithmic axis.

## ReLU, and He's start

For ReLU, $\phi(z) = \max(0, z)$, the gate is 1 where $z > 0$ and 0 elsewhere. A bell centred on 0 is positive half the time, so $\mathbb{E}[\phi'(z)^2] = \tfrac{1}{2}$ and $\chi = \sigma^2/2$. The signal going forward loses the same half: $\mathbb{E}[\phi(z)^2]$ is half the bell's mean square.

Example: LeCun's start has $\sigma^2 = 1$, so the size falls by $\sqrt{1/2} \approx 0.707$ a layer, and from layer 50 back to layer 1, 49 factors, by $0.707^{49} = 2^{-24.5} \approx 4.2 \times 10^{-8}$. He's start, $\sigma^2 = 2$, gives $\chi = 1$; with $\sigma^2 = 4$ the gradient grows by $1.414^{49} \approx 2.4 \times 10^{7}$.

For $\tanh$ the slope falls as the signal grows, so $\chi$ changes along the way and the predicted lines bend.

## A finite network

The prediction is what every draw of the weights does when the layers are very wide. In narrower layers the size takes random steps, larger the narrower they are, and typically drifts a little below the line: the average of the logarithm of a random factor is below the logarithm of its average.

## The whole stack

The Jacobian of layers 2 to 50, $J = \partial h_{50}/\partial h_1$, is a product of one matrix per layer, and $g_1 = J^\top g_{50}$. Its singular values $s_1 \ge s_2 \ge \dots$ are how much it stretches its directions. Their mean square is the squared stretch of a random direction, on average, and $\chi$ controls only that average. Products of many random Gaussian matrices spread their singular values further apart with more factors, so a few directions take nearly all the gradient. With orthogonal weights at gain 1 and the identity as activation, $J$ is itself orthogonal: every $s_i$ is exactly 1. With ReLU a shut unit passes nothing back, so no more of the $s_i$ can differ from 0 than the fewest units open in any of these layers.

# In the picture

- Cyan points: the root mean square of each layer's output $h_l$; the cyan line, its prediction.
- Pink points and line: the same for $g_l$. Above the chart, the factors a layer fitted to the points, beside the lines'.
- The yellow dot is the gain; the white mark is where $\chi = 1$.
- On the right, the $s_i$, largest first; dashed, their root mean square; grey, the scheme chosen before.
- Below, the loss in training; dashed, guessing each class's share.

> Try first "too small a start", then drag the yellow dot to the white mark.

see: backpropmat, actgrad, signal, lyapunov
