# The idea

A network learns only if the gradient of its loss reaches every layer at a size its weights can use. This scene is a bench for seeing whether it does: build a network, measure at each layer how large the signal is going forward and the gradient coming back, and train it beside a second design, $B$, kept for comparison. It needs the chain rule and what a network is; the earlier scenes take each effect on its own.

# The mathematics

## Forward, then back

Write $x_0$ for the data and $x_l$ for what layer $l$ hands on, $x_l = f_l(x_{l-1})$, and $L$ for the loss. Backpropagation starts from $g = \partial L / \partial x$ at the top and hands it down, layer by layer, by the chain rule:

$$ g_{l-1} = J_l^\top g_l $$

where $J_l$ is the layer's Jacobian, the matrix of $\partial x_l / \partial x_{l-1}$. For a plain layer $x_l = \varphi(W_l x_{l-1} + b_l)$ it is $J_l = D_l W_l$, with $D_l$ the diagonal matrix of the activation's slopes $\varphi'$: each layer multiplies the gradient by its weights, transposed, and by its gates.

If the weights are drawn with variance $\text{gain}^2 / n$ ($n$ inputs per unit) and the slopes are near 1, as tanh's are for small signals, each square layer multiplies the gradient's root mean square by about the gain (in the limit of very wide layers, by the gain times the root mean square of the slopes). Gain 0.6 over 19 layers gives $0.6^{19} \approx 6.1 \times 10^{-5}$. The first experiment measures $1.7 \times 10^{-7} / (2.4 \times 10^{-3}) \approx 7.1 \times 10^{-5}$ from layer 20 to layer 1.

## Normalisation and skips

A normalisation rescales what passes through: layer normalisation turns each example into $(x - \bar x)/s$, with $s$ the standard deviation of its entries. So the signal leaves every layer at about the same size. Nothing in that keeps the gradient level: in the scene, thirty ReLU layers with batch normalisation (each unit standardised over the batch) hand back a gradient that grows more than 700 times.

A skip connection adds the layer to its input, $x_l = x_{l-1} + s\, f(x_{l-1})$, so

$$ J_l = I + s\, J_f, \qquad g_{l-1} = g_l + s\, J_f^\top g_l $$

The identity hands the gradient down unchanged; the branch adds to it. "Pre" puts the normalisation inside the branch, $x + f(\operatorname{norm} x)$, and leaves that straight road clear; "post", $\operatorname{norm}(x + f(x))$, puts a normalisation's Jacobian on the road at every layer.

## Stretching

The Jacobian of layers 2 to $L$, the last, is $J = J_L \cdots J_2$. Its singular values $\sigma_1 \ge \sigma_2 \ge \dots$ say how much it stretches each direction: a gradient along one of its directions comes back $\sigma_k$ times as large. Spread over many powers of ten, they usually mean a few directions carry most of it.

## The size of a step

Plain gradient descent moves a weight matrix by $\Delta W = -\eta\, \nabla_W L$, with $\eta$ the learning rate; Adam moves each entry typically by $\eta$ or less, whatever the gradient's size. The scene shows $|\Delta W| / |W|$. Example: a $32 \times 32$ matrix drawn with variance $2/32$ has $|W| \approx \sqrt{1024 \cdot 2/32} = 8$. Adam's first step at $\eta = 0.1$ moves all 1024 entries by 0.1, so $|\Delta W| = 0.1 \cdot 32 = 3.2$: 40 % of the matrix in one step.

# In the picture

- The diagram: linear maps yellow, normalisations purple, the activation cyan, the skip and its sum green.
- Cyan: the root mean square of $x_l$ at each layer; the bars under it, the share of units shut.
- Pink: the root mean square of $g_l$, and the singular values $\sigma_k$.
- Yellow: for each weight matrix, $|\nabla_W L|$ or $|\Delta W| / |W|$.
- Orange: the loss; dashed, that of a network that knows nothing.
- $B$ is grey and dashed everywhere.

> Try first "a deep network that will not train": read the gradient chart, then press "train".

see: gradflow, normgrad, residual, tfgrad
