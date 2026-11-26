# The idea

The scene "backprop1" follows the gradient back through a network small enough to write every number on a wire. A real network is worked out a layer and a batch at a time, as matrix products, and so is its gradient. This scene shows that sweep, forward to the loss and back, block by block. It needs the chain rule, matrix products and the transpose, and for the weights' slopes the rank of a matrix.

# The mathematics

## Rows are examples

A batch of $n$ examples is a matrix with one row per example. Layer $l$ has weights $W_l$, with one row per unit of the layer and one column per unit of the layer before, and a bias $b_l$, a row added to every row. With $A_0 = X$, the data,

$$ Z_l = A_{l-1} W_l^\top + b_l, \qquad A_l = \varphi(Z_l) $$

where the activation $\varphi$ (ReLU, tanh, ...) acts on each entry. The read-out's $Z_3$ goes through a softmax, giving the probabilities $P$, and the loss $L$ is the mean over the examples of $-\ln$ of the probability given to the right class. $Y$ holds the right classes, each row zeros but for a 1.

## What comes back

Write $D_l$ for the slope of $L$ with respect to $Z_l$: a matrix of $Z_l$'s shape, entry $(i, j)$ saying how fast $L$ changes as unit $j$'s value for example $i$ is nudged. For softmax with cross-entropy (scene "lossgrad"),

$$ D_3 = (P - Y)/n $$

Going down a layer is the chain rule twice. Unit $k$ of layer $l$ fed every unit $j$ of the layer above, through the weight $(W_{l+1})_{jk}$, so the slope at its output collects $\sum_j (D_{l+1})_{ij} (W_{l+1})_{jk}$: for the whole batch, the product $D_{l+1} W_{l+1}$. The way forward multiplied by $W_{l+1}^\top$; the way back multiplies by its transpose. Then each entry passes through its unit's gate, the activation's slope at the value the unit had:

$$ D_l = (D_{l+1} W_{l+1}) \odot \varphi'(Z_l) $$

($\odot$ multiplies entry by entry.) Where a ReLU unit was off, its $z$ below 0, its gate is 0 and nothing passes back.

## The weights' slopes

The weight $(W_l)_{jk}$ carries input $k$ into unit $j$, so through example $i$ it changes $L$ at the rate $(D_l)_{ij} (A_{l-1})_{ik}$: what came back to the unit times what went in along the wire. Over the batch,

$$ \frac{\partial L}{\partial W_l} = D_l^\top A_{l-1} = \sum_{i=1}^{n} d_i^\top a_i $$

with $d_i$ and $a_i$ the rows of $D_l$ and $A_{l-1}$ for example $i$. Each term is an outer product, a column times a row: every row of it is a multiple of $a_i$, so its rank is 1 (or 0), and a sum of $n$ of them has rank at most $n$. The bias's slope is the column sums of $D_l$.

One example with $d = (2, -1)$ and $a = (3, 0, 1)$ gives

$$ d^\top a = \begin{bmatrix} 6 & 0 & 2 \\ -3 & 0 & -1 \end{bmatrix} $$

whose second row is $-1/2$ times its first. A zero in $a$ (a unit below that was off) makes a column of zeros; a zero in $d$ (a unit here that was off) makes a row of zeros.

## What it costs

The way back multiplies matrices of the forward sizes, two products a layer instead of one: about twice the work. It needs every $A_l$ and gate kept from the way forward, and it never builds a Jacobian (every output against every input), only its product with a slope, as $D_{l+1} W_{l+1}$ is.

# In the picture

- Cyan: $X$, $A_1$, $A_2$ and $P$; grey: the gates $\varphi'(Z_l)$; orange: the targets $Y$.
- Pink, under them: each $D_l$, and in yellow under it its column sums.
- Yellow: the weights $W_l$, and under each its slope, of the same shape.
- A filled square is a positive number, an outline a negative one, its area the size; the white frame is the step being worked out.
- The table works out the entry picked; the small grids are the outer products; the chart shows the singular values.

> Try first "forwards, then back, a layer at a time"; then click a yellow cell of a weight's slope to see the sum it is.

see: backprop1, chain, lossgrad, lowrank
