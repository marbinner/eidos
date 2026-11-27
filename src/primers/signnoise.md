# The idea

Sign descent, Adam and Muon keep the direction of a gradient and throw away its size: sign descent keeps each coordinate's sign, Muon keeps a weight matrix's singular vectors and pushes every singular value towards 1. Trained on mini-batches, they see the true gradient plus noise, and a step that ignores sizes gives the noise's part as much room as the signal's. How much of such a step still points where the whole data's step would, and what do a larger batch or an average over steps buy? You need the idea of a mini-batch gradient (the mean of $B$ examples' gradients), the bell-shaped normal distribution, and singular values.

# The mathematics

## One coordinate

Over the data, one example's gradient in a coordinate has mean $g$ (the true gradient) and standard deviation $\sigma$. The mean of $B$ independent examples has mean $g$ and standard deviation $\sigma/\sqrt{B}$, and for $B$ not too small it is close to a normal bell. Write $r = g\sqrt{B}/\sigma$, taking $g > 0$. The bell's share above zero is the chance that the sign is right, and the expected sign follows:

$$ P(\text{right sign}) = \Phi(r), \qquad \mathbb{E}\,\operatorname{sign} = 2\Phi(r) - 1 \approx 0.80\,r \ \text{ when } r \ll 1 $$

where $\Phi$ is the share of a standard bell below a point. The coordinate's own noise scale $B_i = \sigma^2/g^2$ is the batch at which $r = 1$. Well below it, a sign step is a full-size step whose sign is nearly a coin toss: its mean grows only as $\sqrt{B}$, and its jitter (standard deviation) $\sqrt{1 - (\mathbb{E}\,\operatorname{sign})^2}$ stays near 1. SGD's step is the batch's gradient itself: its mean is right at every batch, and its jitter, $\sigma/\sqrt{B}$, shrinks.

Example: $g/\sigma = 0.0284$ (so $B_i = 1238$) and $B = 16$ give $r = 0.114$. The sign is right with probability $\Phi(0.114) = 0.545$; the sign step is $0.09 \pm 1.00$ in units of the whole data's step, and SGD's is $1 \pm 8.8$.

## Averaging first

Momentum steps from $m \leftarrow \beta m + (1 - \beta)\,g_B$. With independent batches at fixed weights, $m$ is a weighted mean whose weights $(1 - \beta)\beta^k$ have squares summing to $(1 - \beta)/(1 + \beta)$:

$$ \operatorname{Var} m = \frac{1 - \beta}{1 + \beta}\cdot\frac{\sigma^2}{B} $$

So the average is worth a batch $(1 + \beta)/(1 - \beta)$ times larger: 19 at $\beta = 0.9$. In training the weights move meanwhile: this holds at fixed weights.

Adam divides $m$ by $\sqrt{v}$, where $v$ averages squares of single batches' gradients, about $g^2 + \sigma^2/B$. Its numerator is averaged and its denominator is not, so where noise rules its jitter is cut by $\sqrt{(1 - \beta)/(1 + \beta)}$, about 0.23 at $\beta = 0.9$: a small step, not a full-size coin toss.

## A layer of a network

For a whole layer, let $d$ be a rule's step from a mini-batch and $d^*$ its step from the whole data. The useful part is

$$ u = \frac{\mathbb{E}\,\langle d, d^* \rangle}{|d^*|\,\sqrt{\mathbb{E}\,|d|^2}} $$

For SGD, $\mathbb{E}\,\langle g_B, G\rangle = |G|^2$ and $\mathbb{E}\,|g_B|^2 = |G|^2 + \operatorname{tr}\Sigma/B$, so exactly

$$ u_{\text{SGD}} = \frac{1}{\sqrt{1 + B_{\text{simple}}/B}}, \qquad B_{\text{simple}} = \frac{\operatorname{tr}\Sigma}{|G|^2} $$

Sign descent's useful part averages the expected signs of all the coordinates, the noisy ones counting as much as the clean ones; on this layer it needs a larger batch than SGD's for the same useful part. For Muon no formula is used here: its useful part is measured. One example's gradient of a weight matrix has rank one, so a batch of $B$ has rank at most $B$.

# In the picture

- The bell is one coordinate's batch gradient: white side right sign, red side wrong, grey outline the measured batches.
- Below it, each rule's mean step (solid) and jitter (dashed) against $B$: cyan SGD, purple sign descent, orange Adam; the dashed grey line is $B_i$.
- The network's chart shows $u$ for layer 2 against $B$, dashed with momentum; under it, the coordinates' own noise scales, white below $B$.
- The last chart shows the singular values of the true gradient (white), a batch's gradient (cyan) and its noise (red).

> Try first "a coin toss", then "average first: momentum".

see: minibatch, adam, muon, noisescale
