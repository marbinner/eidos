# The idea

Here SGD with momentum, AdamW and Muon train one small network, from the same starting weights on the same batches, each at its own best step size. The scene asks how their training differs: how fast the loss falls, how wide a range of step sizes each bears, what shape each step has, and what becomes of the weights. Every run was recorded beforehand, with three seeds and two batch sizes. You need the singular value decomposition and the idea of a mini-batch gradient.

# The mathematics

## One matrix, three steps

Take one layer's weight matrix $W$ (32 by 32 for the two middle layers) and the gradient $G$ of a mini-batch's loss with respect to it. Each rule turns the gradients into a step $\Delta W$, scaled by its step size $\eta$:

- SGD with momentum keeps a running sum $B \leftarrow 0.9\,B + G$ and steps $\Delta W = -\eta B$.
- AdamW keeps running averages of each entry's gradient and of its square, and divides the first by the root of the second, entry by entry: each entry moves by about $\eta$, rarely more.
- Muon keeps a running average $M \leftarrow 0.95\,M + 0.05\,G$, blends it with the newest gradient, writes the blend as $U \Sigma V^\top$ and steps along a matrix with the same $U$ and $V$:

$$ \Delta W = -\eta\, U f(\Sigma) V^\top $$

Here $f$ divides each singular value by the blend's Frobenius norm and applies five rounds of the polynomial $3.4445\,x - 4.7750\,x^3 + 2.0315\,x^5$, which pushes it towards 1.

## The shape of a step

Write a step's singular values largest first, $\sigma_1 \ge \sigma_2 \ge \dots$. Its stable rank

$$ \frac{\sigma_1^2 + \sigma_2^2 + \dots}{\sigma_1^2} $$

says roughly how many directions the step moves: 1 for a matrix of rank one, $n$ for an orthogonal $n \times n$ one. Singular values 1, 0.5 and 0.1 give $1 + 0.25 + 0.01 = 1.26$.

A layer's gradient for one point is a rank-one matrix: the gradient at the layer's output times its input, $\delta\,x^\top$. A batch's gradient is the mean of such matrices, so its rank is at most the batch size. Here a few directions dominate: on layer 2 the steps of SGD with momentum, running sums of gradients, have stable ranks below 3. Dividing entry by entry need not flatten that either. In the extreme where each entry's step is only its sign, the signs of a rank-one $u v^\top$ are $\operatorname{sign}(u)\operatorname{sign}(v)^\top$, again of rank one: $u = (1, -2)$ and $v = (3, 1)$ give

$$ \begin{bmatrix} 3 & 1 \\ -6 & -2 \end{bmatrix} \mapsto \begin{bmatrix} 1 & 1 \\ -1 & -1 \end{bmatrix} $$

Muon's $f$ flattens, but only within reach: it brings a singular value to between 0.68 and 1.2 if it is at least about 0.0015 of the Frobenius norm, and leaves smaller ones below 0.68. A momentum whose singular values fall away steeply therefore gives a step that is flat over its strongest directions and smaller beyond them.

## Sizes

The three step sizes are not one scale. Muon's step has its largest singular values near $\eta$, so it changes the layer's output by at most about $1.2\,\eta$ times the input's length. An AdamW step of $\eta$ moves each of the 1024 entries by about $\eta$; if their signs form a rank-one pattern, the step stretches some input by $32\,\eta$ times its length. So the scene compares how wide a range of step sizes each rule bears, not the values. How large a step is beside the weights is measured by

$$ \frac{\|\Delta W\|_F}{\|W\|_F}, \qquad \|A\|_F^2 = \sum_{i,j} a_{ij}^2 = \sum_i \sigma_i^2 $$

# In the picture

- Pink, orange and blue are SGD with momentum, AdamW and Muon in every chart.
- At the top, the network's four weight matrices (click one), and under them each one's $\|\Delta W\|_F / \|W\|_F$ at the step shown.
- The loss against the step; beside it the final loss against $\eta$, each rule's best ringed.
- The chosen layer's step: its singular values over $\sigma_1$, with the stable rank; and its weights' singular values, the start dashed.

> Try first "each rule at its best step", then "the step's shape, early and late".

see: svd, adam, muon, signnoise
