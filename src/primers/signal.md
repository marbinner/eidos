# The idea

Before training, a deep network with random weights already does something to its inputs: layer after layer it makes the activity for two inputs more alike, or less. Which happens decides whether signals, and the gradients that train the network, survive many layers, and for a very wide network it can be read off one curve. You need the playground's layers and activations, and correlation.

# The mathematics

## One random layer

Write $z$ for a layer's activity before its activation $\varphi$, one number per neuron. Each layer has width $n$; the next computes $z' = W \varphi(z) + b$, each weight drawn at random with standard deviation $\sigma_w / \sqrt{n}$ and each bias with $\sigma_b$. Feed in two inputs whose activities have the same length $q$ (the mean square of the entries of $z$) and correlation $c$ (the cosine of the angle between them).

Each entry of $z'$ is a sum of $n$ random terms plus a bias. In the limit of infinite width the central limit theorem makes these entries Gaussian, and averages over neurons become expectations. Then, exactly,

$$ q' = \sigma_w^2\, \mathbb{E}[\varphi(u)^2] + \sigma_b^2, \qquad c' = \frac{\sigma_w^2\, \mathbb{E}[\varphi(u)\varphi(v)] + \sigma_b^2}{q'} $$

where $u$ and $v$ are Gaussian with mean 0, variance $q$ and correlation $c$. The product $\varphi(u)\varphi(v)$ appears because both inputs meet the same weights. When the length settles at a value $q^*$ that the layer keeps, the second formula is one fixed curve, $c' = M(c)$, with $M(1) = 1$.

## Order, chaos and the edge

Depth applies $M$ again and again (the staircase), and near $c = 1$ its slope decides:

$$ \chi = M'(1) = \sigma_w^2\, \mathbb{E}[\varphi'(u)^2] $$

Near 1, each layer multiplies the gap to perfect correlation by $\chi$:

$$ 1 - c' \approx \chi\,(1 - c) $$

If $\chi < 1$ the gap shrinks and inputs end up alike (order). If $\chi > 1$ even a tiny difference grows, until $c$ comes to rest below 1 (chaos). $\chi = 1$ is the edge. In the same theory, $\chi$ is also the factor by which a gradient's mean square changes at each layer on the way back: off the edge, gradients die out or blow up with depth.

## Powers of $c$

Expand the activation in the Hermite polynomials $h_0 = 1$, $h_1 = x$, $h_2 = (x^2 - 1)/\sqrt{2}$ and so on: $\varphi(\sqrt{q^*} x) = \sum_k a_k h_k(x)$. Their one useful property, for standard Gaussians $X$ and $Y$ with correlation $c$, is

$$ \mathbb{E}[h_j(X)\, h_k(Y)] = c^k \text{ if } j = k, \quad 0 \text{ otherwise} $$

Put into the formula for $c'$, it gives

$$ M(c) = \sum_{k \ge 0} s_k\, c^k, \qquad s_k = \frac{\sigma_w^2\, a_k^2}{q^*} $$

with $\sigma_b^2 / q^*$ added to $s_0$. These are the bars. They are never negative and add up to $M(1) = 1$, so they are shares, and $\chi = \sum_k k\, s_k$ is their average power. With $\varphi = h_2$, $\sigma_w = 1$, $\sigma_b = 0$ and $q^* = 1$ there is one bar, $s_2 = 1$, and $M(c) = c^2$.

Example: ReLU, $\varphi(u) = \max(0, u)$, with $\sigma_w = \sqrt{2}$ and $\sigma_b = 0$. By symmetry $\mathbb{E}[\max(0, u)^2] = q/2$, so $q' = 2 \cdot q/2 = q$: every length is kept. The slope $\varphi'$ is 1 half the time and 0 otherwise, so $\chi = 2 \cdot \tfrac{1}{2} = 1$: the edge. Yet unrelated inputs leave one layer at $M(0) = 1/\pi \approx 0.32$, so the curve lies above the diagonal and slowly pulls inputs together.

The white dots are one real network of finite width (512 at first). They scatter round the theory, less for a wider network, and leave it where the balance of lengths is unstable (a small error in $q$ grows).

# In the picture

- The cyan curve is $M$, the dashed line $c' = c$; the yellow staircase is your two inputs, and the pink point is where correlations come to rest.
- The chart is $c$ layer by layer: the line from $M$, the dots from the real network.
- The diagram of all settings colours each $(\sigma_w, \sigma_b)$ by $\chi$: blue order, red chaos, grey where lengths grow without bound.
- The purple bars are the shares $s_k$.

> Try first "order: small weights", then "chaos: large weights".

see: mlp, hermite_orth, hermite_expand, lyapunov
