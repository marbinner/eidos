# The idea

An attention head moves information along a sequence: each position takes a weighted average of what it and the positions before it offer. This scene follows the gradient back through one head. It arrives at one position's output and splits into two roads: through the values, which teaches the head what to move, and through the attention weights, which teaches it where to look. You need the chain rule, the dot product, and the softmax (scene lossgrad).

# The mathematics

## The forward pass

Each position $j$ holds a token $x_j$, from which the head makes a query $q_j = W_Q x_j$, a key $k_j = W_K x_j$ and a value $v_j = W_V x_j$. Position $i$ scores every $j \le i$ by $s_{ij} = q_i \cdot k_j / \sqrt{d}$, with $d$ the head's width (2 here), weighs the values by a softmax of the scores, and outputs the mix

$$ a_{ij} = \frac{e^{s_{ij}}}{\sum_{l \le i} e^{s_{il}}}, \qquad o_i = \sum_{j \le i} a_{ij}\,v_j $$

The weights are positive and add up to 1: $o_i$ is an average of the values. Positions after $i$ are left out: the causal mask.

## The value road

Let $g = \partial L/\partial o_i$ be the gradient of the loss $L$ at the output, sent back from above. The value $v_j$ enters $o_i$ only through $a_{ij} v_j$, so

$$ \frac{\partial L}{\partial v_j} = a_{ij}\,g $$

Each value receives a copy of $g$ scaled by the attention paid to it, and the shares add up to $g$. From there it goes on through $W_V^\top$ to the token, and to $W_V$.

## The pattern road

Hold the values fixed: the slope for a weight is $\partial L/\partial a_{ij} = g \cdot v_j$, negative where more of $v_j$ would lower the loss. The weights come from the scores through the softmax, whose Jacobian is

$$ \frac{\partial a_{ij}}{\partial s_{il}} = a_{ij}\,(\delta_{jl} - a_{il}) $$

where $\delta_{jl}$ is 1 if $j = l$, else 0. By the chain rule,

$$ \frac{\partial L}{\partial s_{ij}} = a_{ij}\,(g \cdot v_j - g \cdot o_i) $$

A step of gradient descent raises the score of every source whose value is better than the mix ($g \cdot v_j < g \cdot o_i$) and lowers those that are worse, each by its weight times how far it is from the mix. The slopes add up to 0: adding one number to every score changes no weight. From the scores it goes on to the query of $i$ and the keys.

## A head that is sure

If one weight is near 1, every entry of the Jacobian is near 0: the pattern road closes, however much better another source would be. The value road stays open: the source looked at receives nearly all of $g$.

Example: two sources with $g \cdot v_1 = -1$ and $g \cdot v_2 = 1$. With even weights, $1/2$ each, $g \cdot o_i = 0$ and the scores' slopes are $-1/2$ and $+1/2$. Sure of the worse source, with $a_{i1} = 0.01$ and $a_{i2} = 0.99$, $g \cdot o_i = 0.98$ and the slopes are $0.01 \times (-1.98) = -0.0198$ and $0.99 \times 0.02 = +0.0198$: about 25 times smaller.

## One token, three roles

A token is read three times, so its gradient is a sum of three: as a value, $W_V^\top\,\partial L/\partial v_j$; as a key, $W_K^\top\,\partial L/\partial k_j$; and, for the destination itself, as the query, $W_Q^\top\,\partial L/\partial q_i$. Nothing reaches a token after $i$.

# In the picture

- The ringed token is the destination $i$, where $g$ arrives. Cyan arcs over them: its attention $a_{ij}$.
- Pink arcs under them: the value road, as wide as each share. Light pink bars: the pattern road, $-\partial L/\partial s_{ij}$, up where a step of descent raises the score.
- The plane: the values $v_j$, their mix $o_i$ and $g$ (drag its tip). The dashed line holds the values exactly as good as the mix.
- The cards: the attention grid, the softmax's Jacobian, the chart of both roads against $\beta$, and a token's three roles.

> Try first "the pattern road: look more at what would help", then "a head that is sure stops learning where to look".

see: chain, lossgrad, backpropmat, tfgrad
