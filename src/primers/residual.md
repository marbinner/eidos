# The idea

Backpropagation hands the gradient of the loss down a network, multiplying it at each block by the block's transposed Jacobian. Through fifty plain layers fifty factors multiply: if each shrinks the gradient a little, or grows it, they can leave almost nothing, or far too much. A skip connection adds each block's input to its output. On the way back that turns the product into a sum over paths, one of which goes round every block. You need the chain rule for vectors (scene chain) and what a network layer computes (scene mlp).

# The mathematics

## One block

Write $x_l$ for the stream after block $l$ and $g_l$ for the gradient of the loss there. A block with a skip connection computes

$$ x_l = x_{l-1} + s\,f_l(x_{l-1}) $$

with $f_l$ the block's own work (here one layer with a ReLU), scaled by $s$. Its Jacobian is $I + J_l$, with $J_l$ the Jacobian of $s\,f_l$, so by the chain rule

$$ g_{l-1} = (I + J_l)^\top g_l = g_l + J_l^\top g_l $$

The gradient reaches $x_{l-1}$ twice: unchanged along the skip, and through the block.

## A sum over paths

Through $L$ blocks the factors multiply:

$$ g_0 = (I + J_1)^\top (I + J_2)^\top \cdots (I + J_L)^\top g_L $$

Multiplied out, each factor contributes $I$ (round block $l$) or $J_l^\top$ (through it): $2^L$ terms, one for each path. For two blocks,

$$ (I + J_1)^\top (I + J_2)^\top = I + J_1^\top + J_2^\top + J_1^\top J_2^\top $$

The term that goes round every block is $g_L$ itself: the straight road. Whatever the blocks do, it is there.

Example: in one dimension each Jacobian is a number. Take three blocks of slope $0.5$: $(1 + 0.5)^3 = 3.375 = 1 + 1.5 + 0.75 + 0.125$. The straight road brings 1, the three paths through one block $3 \times 0.5$, the three through two $3 \times 0.25$, the one through all three $0.125$. Three plain layers would hand back only that last term.

A path through $k$ blocks picks up $k$ Jacobians, so when they are small each long path brings very little. Whether the long paths together bring little depends also on how many there are: in the scene's ten blocks the paths through at most two blocks bring nearly all of $g_0$ at $s = 1/\sqrt{10}$, and not at $s = 1$. It is a fact about a network at a scale, not a law.

## Shares

In a network the terms are vectors pointing different ways. They add up to $g_0$, so each term $t$ is given its share

$$ \frac{t \cdot g_0}{|g_0|^2} $$

and the shares add up to 1. A term can be long and bring little, if it points mostly across $g_0$.

## The price, and pre and post

Going forward every block adds to the stream. With $s = 1$, each block of the scene's network adds about as much as the stream holds; with $s = 1/\sqrt{L}$ the fifty blocks leave it 6.8 times its starting size, most of that a shared offset, since a ReLU never outputs a negative number.

A normalisation $N$ can go inside the block, $x_l = x_{l-1} + s\,f_l(N(x_{l-1}))$ ("pre"), or after the sum, $x_l = N(x_{l-1} + s\,f_l(x_{l-1}))$ ("post"). In pre the road stays clear. In post every step of it is multiplied by $N_l^\top$; for layer normalisation with its gain at 1, $y$ the standardised sum and $\sigma$ its spread,

$$ N^\top h = \frac{h - \operatorname{mean}(h) - y \operatorname{mean}(h\,y)}{\sigma} $$

which removes the parts of $h$ along the all-ones direction and along $y$, and divides the rest by $\sigma$. In the scene's network with $s = 1$, fifty such steps leave the road $1/126$ of what arrived.

# In the picture

- Three blocks: the road (green) with the blocks below it, and the eight paths, green round a block and pink through one. Bars: each path's share of $g_0$; outlines: its term's length over $|g_0|$.
- Every layer: the root mean square of $x_l$ (cyan), $g_l$ (pink) and the straight road's part of $g_l$ (green); solid with skips, dashed without.
- Ten blocks: $g_0$ by the number of blocks passed.
- Training: the loss of both, orange.

> Try first "fifty plain layers, and fifty with skips", then "three blocks are eight paths".

see: chain, gradflow, normgrad, expsum
