# The idea

A network's loss is an average over its training examples, so its gradient is an average too: of one gradient per example. A step of stochastic gradient descent (SGD) averages only a few of them, a mini-batch. Its gradient is right on average and off by noise, and the noise shrinks as the batch grows. Here we work out how much of it is signal and how much noise, and the batch size at which the two are equal. You need the gradient, mean and variance, and vectors with many coordinates.

# The mathematics

## One gradient per example

With $N$ examples and the loss $L = \frac1N \sum_n \ell_n$, the gradient is the mean of the examples' own gradients:

$$ G = \frac1N \sum_{n=1}^{N} g_n, \qquad g_n = \nabla \ell_n $$

Each $g_n$ has one number per weight, 371 here. How they spread round $G$ is their covariance

$$ \Sigma = \frac1N \sum_{n=1}^{N} (g_n - G)(g_n - G)^\top $$

Its trace, the sum of its diagonal, is $\operatorname{tr}\Sigma = \frac1N \sum_n |g_n - G|^2$: the mean squared distance of an example's gradient from $G$.

## A mini-batch

A mini-batch gradient is the mean of $B$ of the $g_n$. Write it $g_B = G + \xi$, with $\xi$ the noise. If the $B$ examples are drawn independently, each from all $N$ (with replacement), each one's gradient has mean $G$ and covariance $\Sigma$, and the mean of $B$ independent terms has $1/B$ of the covariance of one:

$$ \mathbb{E}\,\xi = 0, \qquad \operatorname{Cov}(\xi) = \frac{\Sigma}{B} $$

So the noise's spread, a standard deviation, falls as $1/\sqrt{B}$. If instead the batch is $B$ different examples (without replacement, as an epoch deals them out), the covariance is smaller for every $B$ above 1:

$$ \operatorname{Cov}(\xi) = \frac{N - B}{B\,(N - 1)}\,\Sigma $$

At $B = N$ that is 0: the batch is the whole data set, and its gradient is $G$.

## Signal and noise

Take squared lengths. The cross term $2\,G \cdot \xi$ has mean 0, and the mean of $|\xi|^2$ is the trace of its covariance, so with replacement

$$ \mathbb{E}\,|g_B|^2 = |G|^2 + \frac{\operatorname{tr}\Sigma}{B} $$

The signal $|G|^2$ does not depend on $B$; the noise falls as $1/B$. They are equal at the simple noise scale

$$ B_{\text{simple}} = \frac{\operatorname{tr}\Sigma}{|G|^2} $$

a number of examples. Well below it a mini-batch gradient is mostly noise; well above it, it is nearly $G$. Since $\mathbb{E}\,G \cdot g_B = |G|^2$, replacing $|g_B|$ by the root of its mean square gives an approximate cosine between $g_B$ and $G$ (not a bound either way):

$$ \cos(g_B, G) \approx \frac{1}{\sqrt{1 + B_{\text{simple}}/B}} $$

Example: at the start, $|G|^2 = 0.991$ and $\operatorname{tr}\Sigma = 5.856$, so $B_{\text{simple}} = 5.91$. A batch of 6 has noise $5.856/6 = 0.976$, nearly the signal; a batch of 59 has $0.0993$, a tenth of it.

## Seeing 371 numbers in two

A picture shows a plane: the window is spanned by $e_1 = G/|G|$ and by $e_2$, the direction at right angles to $G$ in which the $g_n$ spread most. Each gradient is drawn at $(g \cdot e_1,\ g \cdot e_2)$. The map is linear, so a batch's dot is the mean of its examples' dots. It shows only part of $\operatorname{tr}\Sigma$; the charts use all 371 coordinates.

## It changes as the network learns

$B_{\text{simple}}$ belongs to one set of weights. In this scene's run of SGD, $|G|^2$ became about 30 times smaller while $\operatorname{tr}\Sigma$ fell by a third, so the noise scale grew about twentyfold.

# In the picture

- The white arrow is $G$; the grey dots are the $g_n$; the red dots are forty mini-batch gradients $g_B$, and the red ring round the tip of $G$ is their covariance $\Sigma/B$ in the window, drawn two standard deviations out.
- On the chart, logarithmic both ways, the white line is $|G|^2$ and the red line $\operatorname{tr}\Sigma/B$ (dashed: the other way of drawing); they cross at $B_{\text{simple}}$. The red dots are each batch's $|g_B - G|^2$, the rings their mean.
- Below, the noise scale along a run of SGD, and its two parts.

> Try first "one example at a time", then drag the red dot on the chart to the right and watch the batches close in on the tip of $G$.

see: backpropmat, clt, noisefloor, noisescale
