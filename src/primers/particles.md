# The idea

Here a network has one wide hidden layer, each neuron drawn as a dot placed by its two input weights. A layer can fit the data in two ways: by moving its neurons to where they are useful (feature learning), or by leaving them almost where chance put them and spreading tiny changes over all of them (the lazy regime). One number, the output scale $\alpha$, decides which. You need the playground's neurons, loss and gradient descent.

# The mathematics

## The layer and its output scale

There are $N$ neurons (200 at first) on the inputs $x = (x_1, x_2)$. Neuron $j$ has input weights $w_j = (w_{j1}, w_{j2})$, a bias $b_j$ and an output weight $a_j$, and with the activation $\varphi$ (ReLU at first) the layer computes

$$ g(x) = \frac{1}{\sqrt{N}} \sum_{j=1}^{N} a_j\, \varphi(w_j \cdot x + b_j) $$

The output, whose sign says orange or blue, is

$$ f(x) = \alpha \big( g(x) - g_0(x) \big) $$

where $g_0$ is $g$ at the starting weights, so that $f$ starts at 0 whatever $\alpha$ is. Training is gradient descent on the playground's loss with the step $\eta / \alpha^2$ (shortened if needed for stability).

## Why $\alpha$ decides

Write $\theta$ for all the $w_j$, $b_j$ and $a_j$. The gradient of the loss carries the factor $\alpha$ from $f$; times the step $\eta / \alpha^2$, each step moves $\theta$ in proportion to $1/\alpha$. To first order the output changes by $\alpha$ times the gradient of $g$ times that move, in proportion to $\alpha \cdot (1/\alpha)$: the same for every $\alpha$, at the start.

Example: at the start, for the same change in $f$, the weights move $1/20$ as far at $\alpha = 20$ as at $\alpha = 1$, and 20 times as far at $\alpha = 0.05$: $20 / 0.05 = 400$ times as far in "feature learning" as in "lazy".

## Lazy: exact in a limit

While $\theta$ stays near its start $\theta_0$, the first-order Taylor expansion holds:

$$ f(x) \approx \alpha\, \nabla g_0(x) \cdot (\theta - \theta_0) $$

with $\nabla g_0(x)$ the gradient of $g$ with respect to $\theta$ at the start. This is linear in $\theta$: fixed random features with adjustable weights. It is a theorem that as $\alpha \to \infty$, with steps scaled as here and small, and a smooth activation, training follows this linear model ever more closely over any fixed stretch of training. As $N \to \infty$ with $\alpha$ fixed and small steps the same happens (smooth activations again), since the $1/\sqrt{N}$ in $g$ makes each neuron move less and less.

A finite layer is only near the limit: at $N = 200$ and $\alpha = 20$ the weights move a few per cent. As a rule of thumb, in lazy training the share moved goes as $1/(\alpha \sqrt{N})$, so 16 neurons instead of 200 should move about $\sqrt{200/16} \approx 3.5$ times as much, near what "lazy, with only 16 neurons" shows.

## Feature learning: the other limit

For small $\alpha$ the weights must travel far, the expansion fails, and the features themselves change: the neurons gather where this data needs them. The natural small value is $\alpha = 1/\sqrt{N}$, which makes $f$ an average over the neurons:

$$ f(x) = \frac{1}{N} \sum_{j=1}^{N} a_j\, \varphi(w_j \cdot x + b_j) \; - \; (\text{its value at the start}) $$

It is proved, under conditions on the activation, that as $N \to \infty$ with this scaling and small steps the cloud of dots moves by a fixed law and a neuron travels as far whatever $N$ is. For $N = 200$, $1/\sqrt{200} \approx 0.071$, close to the 0.05 of "feature learning". The scene shows one finite network, between the two limits.

# In the picture

- Left: a dot is neuron $j$ at $w_j$. Orange means $a_j > 0$, blue $a_j < 0$, and a bigger dot a larger $|a_j|$. The grey dot is where it started; the thread is its path.
- Right: the data and the answer; the white line is $f = 0$. Each faint line is where $w_j \cdot x + b_j = 0$, where a ReLU neuron switches on.
- "Weights moved" is $|\theta - \theta_0| / |\theta_0|$.

> Try first "feature learning ($\alpha = 0.05$)", then "lazy ($\alpha = 20$)".

see: mlp, fit1d, landscape
