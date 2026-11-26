# The idea

A transformer's loss at one position depends on every weight, and on the residual stream at every layer, at that position and earlier ones. This scene follows the gradient of that loss back through the network: down the stream, and across the positions. Does it reach the bottom, which positions does it reach, and how much reaches each part's weights? It helps to know the chain rule and what a transformer block does: attention, then an MLP.

# The mathematics

## Down the stream

At each position the stream is a row $x$ of 32 numbers. A pre-norm block, with $\operatorname{LN}$ the layer normalisation, is

$$ m = x + \operatorname{attn}(\operatorname{LN}(x)), \qquad x' = m + \operatorname{mlp}(\operatorname{LN}(m)) $$

Write $g_x$ for the gradient of the loss $L$ with respect to $x$. The chain rule through $m = x + f(x)$ gives

$$ g_x = g_m + J_f^\top g_m $$

where $J_f$ is the Jacobian of the part $f$. The first term, the skip road, hands the gradient from the place above down unchanged; the second came back through the part. If $g_m = (1, 2)$ and the part hands back $(0.5, -1)$, then $g_x = (1.5, 1)$. In pre-norm the gradient at the bottom is the one at the top plus one such term for every part above, so it need not shrink with depth.

In post-norm, $x' = \operatorname{LN}(x + f(x))$, and every step down the road passes through the normalisation. With its gain at 1, as at the start, its Jacobian takes two directions out of the gradient (the all-ones direction and that of its normalised input) and divides the rest by the spread $\sigma$ of the sum it normalised. Where $\sigma$ is above 1, each step down the road shrinks the gradient. Without skip connections there is no road: all of $g_x$ comes through the parts.

## Across the positions

The MLP and the layer normalisation work on each position by itself, so the gradient at position $s$ comes only from position $s$ above. Attention mixes positions: a head's output at position $i$ is $o_i = \sum_{j \le i} a_{ij} v_j$, a mix of earlier values $v_j$ with weights $a_{ij}$ that add up to 1. So

$$ \frac{\partial L}{\partial v_j} = \sum_{i \ge j} a_{ij} \, \frac{\partial L}{\partial o_i} $$

and more reaches position $j$ through the weights $a_{ij}$ themselves. Only $j \le i$ appear: nothing reaches a later position. If position 3 looks at positions 1, 2, 3 with weights 0.5, 0.3, 0.2, the value of position 1 receives half of the gradient arriving at position 3's output, and position 4 receives nothing.

## Small where it has learnt

The loss at a position is $L = -\ln p_y$, with $p$ the network's chances for the next token and $y$ the right one. Its gradient with respect to the scores the chances come from is $p - e_y$, where $e_y$ has 1 at $y$ and 0 elsewhere. With three tokens and even chances, $p - e_y = (-2/3, 1/3, 1/3)$, of length 0.82; with chances $(0.98, 0.01, 0.01)$ it is $(-0.02, 0.01, 0.01)$, of length 0.024. Every other gradient is this vector carried back by the chain rule, a linear map for a given network, so they all shrink with it.

# In the picture

- The cells: the length of $g$ at each place of the stream (rows) and position (columns), on a logarithmic scale; an empty cell is exactly 0.
- Orange: the position whose loss is $L$; the thin bar under a next token is its chance $p_y$.
- The yellow bars: the length of the gradient reaching each head's and each MLP's weights $W$; beside each, its relative step $\eta \, |\partial L / \partial W| / |W|$ at the training rate $\eta = 0.004$.
- The chart: the length of $g$ at the loss's position, from the embeddings (block 0) up; dashed, a design kept for comparison.
- One cell taken apart: green, the skip road; blue, what came through the part above. The top strip is their sum, number by number.

> Try first "where one loss's gradient goes", then "take the skips away".

see: chain, residual, attngrad, normgrad
