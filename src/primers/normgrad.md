# The idea

A deep network passes a signal forward through many layers, and a gradient back. If each layer changes the size of what passes through it, the sizes multiply, and the signal or the gradient dies away or blows up. A normalisation layer rescales what passes through it to a standard size. This scene shows what that does going forward, and the less familiar thing it does going back. It assumes vectors, and backpropagation through a layer's Jacobian.

# The mathematics

## Three normalisations

For one example with $n$ units, $x = (x_1, \dots, x_n)$, the root mean square is

$$ \operatorname{rms}(x) = \sqrt{(x_1^2 + \dots + x_n^2)/n} = |x| / \sqrt{n} $$

- RMS normalisation: $y = x / \operatorname{rms}(x)$. Nothing is taken off.
- Layer normalisation: the mean of the units is taken off, the rest divided by their standard deviation.
- Batch normalisation: the same for each unit, over the examples of the batch.

Scale $x$ by any $c > 0$ (for batch normalisation, every example's) and $y$ does not change.

## The gradient on the way back

Let $g = \partial L / \partial y$ be the gradient of the loss $L$ arriving at $y$, and $u = x / |x|$ the direction of $x$. For RMS normalisation

$$ \frac{\partial L}{\partial x} = \frac{g - (g \cdot u)\,u}{\operatorname{rms}(x)} $$

Two facts give it. A step along $x$ does not move $y$, so the loss has no slope that way: $g$'s part along $x$ is dropped. A small step $h$ at right angles to $x$ turns its direction by $h / |x|$, and so moves $y = \sqrt{n}\,u$ by $h / \operatorname{rms}(x)$: the rest is divided by $\operatorname{rms}(x)$. So what comes back is at right angles to $x$, and halves when $x$ doubles. Layer normalisation does the same after taking the mean off, so its gradient loses its mean.

Example, the scene's first picture: $x = (0.76, 0.38)$, so $|x| = 0.850$, $\operatorname{rms}(x) = 0.601$ and $u = (0.894, 0.447)$. With $g = (0.25, 0.95)$, $g \cdot u = 0.648$, and the rest is $g - 0.648\,u = (-0.330, 0.660)$, of length $0.738$. Divided by $0.601$ it is $(-0.549, 1.098)$, of length $1.23$.

## Weights in front of a normalisation

If weights $W$ feed a normalisation with no bias between them, $\operatorname{norm}(cWx) = \operatorname{norm}(Wx)$ for every $c > 0$, so the loss does not depend on the size of $W$. By the same two facts, $\nabla_W L$ is at right angles to $W$, and $1/c$ times as long when $W$ is $c$ times as long. A step of gradient descent at a fixed rate $\eta$ therefore turns $W$ by an angle $\theta$ with

$$ \tan\theta = \eta\,|\nabla_W L| / |W| $$

which goes as $1/c^2$: a larger weight learns more slowly.

## Many layers, and the batch

Without a normalisation, the gain of the starting weights decides whether each layer of a plain network enlarges or shrinks the signal; over 20 layers the sizes grow or fade geometrically. After each normalisation the signal has the same size whatever the gain. In the scene's network the gradient stays within a factor of 4 of level with layer or RMS normalisation; with batch normalisation it grows by about 1.3 to 1.4 a layer on its way back.

Batch normalisation divides by statistics of the whole batch, so one example's output and gradient depend on the others. That is in training; afterwards the statistics are usually fixed, and each example is normalised on its own.

# In the picture

- First: $x$ (cyan), $y$ on the purple circle $\operatorname{rms}(y) = 1$, the arriving $g$ (pink) with its part along $x$ dashed, and $\partial L/\partial x$ at $x$.
- Second: five examples $x_i$ and their outputs $y_i$; $g$ arrives at $y_1$ alone, and comes back to every $x_i$.
- Third: the root mean square of each layer's output (cyan) and gradient (pink), solid with the normalisation, dashed without.
- Fourth: layer 10's weights $W$ times $c$ (yellow), and a step of gradient descent (pink).

> Try first "what comes back has lost its part along the input", then "a deep network at a bad gain, rescued".

see: backpropmat, gradflow, residual
