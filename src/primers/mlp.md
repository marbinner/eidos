# The idea

A neural network is a function built from pieces called neurons. Here it takes a point $x = (x_1, x_2)$ and answers how likely it is to be orange. Training adjusts its weights by a form of gradient descent until the answers fit a set of coloured points. You need vectors, matrices, derivatives and gradient descent.

# The mathematics

## Neurons and layers

A neuron weighs its inputs, adds a bias $b$, and passes the sum $z$ through a fixed bend $\varphi$, the activation:

$$ a = \varphi(z), \qquad z = w_1 x_1 + w_2 x_2 + b $$

On the plane $z$ is constant along straight lines, so a first-layer neuron sees only which side of a straight edge a point is on, and how far. Two of the activations: $\tanh$, a smooth step from $-1$ to $1$, and ReLU, $\varphi(z) = \max(0, z)$.

A layer is several neurons reading the same inputs. With their weights as the rows of a matrix $W$ and their biases in a vector $b$, it turns the activity $a$ of the layer before into

$$ a' = \varphi(W a + b) $$

with $\varphi$ applied to each entry. Each hidden layer reads the one before, so later neurons combine straight edges into curves. The scene opens with two hidden layers of 8: $2 \cdot 8 + 8$, $8 \cdot 8 + 8$ and $8 + 1$ weights and biases, the 105 parameters the panel counts.

## The answer and the loss

The output neuron has no bend. Its value $s$ becomes a probability of orange:

$$ p = \frac{1}{1 + e^{-s}} $$

so $s = 0$ is even odds. For a point with label $t$ (1 for orange, 0 for blue) the loss is

$$ L = -t \ln p - (1 - t) \ln(1 - p) $$

averaged over the points: minus the logarithm of the probability given to the right answer. It is near 0 when the network is sure and right, and $\ln 2 \approx 0.69$ at even odds.

## Backpropagation

Gradient descent needs the derivative of $L$ with respect to every weight. The network is a chain of simple functions, so the chain rule builds each derivative from local slopes: multiplied along each path back from the loss, added over paths. At the output,

$$ \frac{\partial L}{\partial s} = p - t $$

A neuron that feeds $s$ through the weight $v$ receives this error times $v$; one that feeds several neurons adds up what it receives from each. It multiplies that by its own slope $\varphi'(z)$, and a weight that carries a value $x_k$ into it has derivative equal to the result times $x_k$. One sweep of these moves, from the output back to the inputs, finds every derivative: backpropagation.

Example: a ReLU neuron with weights $(2, -1)$ and bias $0.5$ feeds the output with $v = 2$, and the output's bias is $-3$. At the orange point $x = (1, 1)$: $z = 1.5$, $a = 1.5$, $s = 2 \cdot 1.5 - 3 = 0$, $p = 0.5$, $L = \ln 2$. Backwards: $p - t = -0.5$, times $v = 2$ is $-1$, times the slope 1 (the ReLU is on) is $-1$, times $x_1 = 1$ gives $\partial L / \partial w_1 = -1$. A step with learning rate 0.1 moves $w_1$ to 2.1, which alone raises $s$ to 0.2, towards orange.

Each step does this for a batch of points (32 at first), averages, and moves every weight by a rule called an optimiser (Adam here). Held-out points are never trained on: when their loss rises while the training loss falls, the network is fitting the noise.

# In the picture

- The big square is $p$ at every place: blue for blue, orange to red for orange, pale near even odds. The white line is the boundary, $s = 0$.
- Each tile is one neuron's output $a$ over the same plane, red high and blue low. A line between tiles is a weight: orange positive, blue negative, thicker for larger.
- The chart shows the loss on training points (cyan) and held-out points (orange).

> Try first "Two neurons cannot close a ring", then raise "neurons per layer" to 3.

see: surface, descent, fit1d, fold
