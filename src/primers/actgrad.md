# The idea

To train a network, the gradient of its loss is passed backwards, layer by layer, to every weight. On the way it goes through every unit's activation, and there, by the chain rule, it is multiplied by the activation's slope at the input the unit had on the way forward. That slope is the unit's gate. This scene compares the gates of nine activations, and what they do in a deep network and in a network that learns. It needs the chain rule (scene "chain") and what a network's layer computes (scene "mlp").

# The mathematics

## One gate

A unit turns its input $z$ into its output $a = f(z)$. When the gradient of the loss $L$ comes back to $a$, the chain rule gives

$$ \frac{\partial L}{\partial z} = f'(z)\,\frac{\partial L}{\partial a} $$

At a gate of 1 the gradient goes through whole, at 0 nothing does, and below 0 it comes back with its sign turned round. A gate smaller than 0.01 in size is called shut.

- The sigmoid $1/(1 + e^{-z})$ has slope at most $1/4$, at $z = 0$.
- tanh has slope 1 at $z = 0$, but below 0.01 once $|z| > 2.99$: saturated.
- ReLU, $\max(0, z)$, has slope exactly 1 above 0 and exactly 0 below: open or shut.
- Leaky ReLU has slope 0.01 below 0, and ELU slope $e^z$: small, but never 0.
- GELU, $z\,\Phi(z)$ with $\Phi$ the normal distribution function, has slope a little below 0 for $z < -0.75$ and a little above 1 for $z > 0.75$.

## Averages over a bell

A unit gets a different $z$ for every example. Suppose these are spread like a bell, a normal distribution with centre $\mu$ and width $\sigma$, and write $\mathbb{E}$ for the average over it. A gradient's size is measured by its mean square, the average of the squares. If what arrives at the outputs is unrelated to the gates, as it nearly is in a wide network with random weights, the gates multiply the mean square by

$$ \mathbb{E}\,f'(z)^2 $$

Example: ReLU with $\mu = 0$. Half the inputs are above 0 (gate 1) and half below (gate 0), so $\mathbb{E}\,f'(z)^2 = 1/2$: the mean square is halved, and its square root, the typical size, is multiplied by $\sqrt{1/2} \approx 0.71$. For the sigmoid with $\mu = 0$, $\sigma = 1$, $\mathbb{E}\,f'(z)^2 = 0.045$: the mean square is divided by about 22.

## Through many layers

Going back through a layer, the gradient is multiplied by the layer's weights (transposed) and then by its gates. In layers as wide as their inputs, weights drawn with variance $1/n$, for $n$ inputs, keep the mean square on average, so each layer multiplies it by its own $\mathbb{E}\,f'(z)^2$. Ten ReLU layers give

$$ \left(\tfrac{1}{2}\right)^{10} = \tfrac{1}{1024} $$

of the mean square, $1/32 \approx 0.031$ of the typical size. The scene's ten layers of 32 units measure 0.027; the rule is exact only for very wide layers.

## Dead units

A unit whose gate is shut for every example is dead. Every term of its weights' gradients has its gate as a factor, so they are 0, and plain gradient descent leaves those weights as they are. In the first layer, whose inputs are the data, a dead unit stays dead for good.

# In the picture

- Cyan goes forward: the activation $f(z)$, the bell of inputs with dots at its centre $\mu$ and width $\sigma$ to drag, and the mean output $\mathbb{E}\,f(z)$, dashed.
- Pink comes back: the gate $f'(z)$, and the histogram of the gates the bell's inputs meet (grey where shut), with both averages written under it.
- The ten layers: the gradient's typical size at each layer, one curve per activation.
- The map: a row for each unit of the network that learns, a column for each point: pink open, black shut, blue below 0.

> Try first "ReLU is open or shut, and can die": most units start shut for every point, and the network learns nothing.

see: chain, mlp, backpropmat, gradflow
