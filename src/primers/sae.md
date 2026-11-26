# The idea

A network can hold more features than it has dimensions by giving them overlapping directions (the Superposition scene). Here we are handed only its activations, points in $m$ dimensions, and must find those directions. A sparse autoencoder rewrites every activation as a sum of a few arrows from a list it learns, and is charged for every unit of arrow it uses. That charge is what lines the list up with the features. You need dot products, lengths of vectors, and training to lower a loss.

# The mathematics

## The activations

There are $n$ true features, each a direction $u_i$ of length 1. An activation is

$$ h = x_1 u_1 + x_2 u_2 + \cdots + x_n u_n $$

(plus noise, if the slider asks for it), where each $x_i$ is $0$ with probability $S$, the sparsity, and otherwise a random number between 0 and 1. On average $n(1 - S)$ features are present: $0.5$ for five features at $S = 0.9$. The autoencoder sees $h$, never $u_i$ or $x_i$.

## The autoencoder

It has $k$ latents. Latent $j$ listens along a direction $e_j$ and writes back along its arrow $d_j$, of length 1:

$$ f_j = \operatorname{ReLU}\big(e_j \cdot (h - b) + c_j\big), \qquad \hat h = b + \sum_j f_j d_j $$

with $b$ an offset shared by all and $c_j$ an offset of the latent's own. The latent fires when $f_j > 0$. Training makes this small, on average over activations:

$$ \lVert h - \hat h \rVert^2 + \lambda \sum_j f_j $$

The first term is the error of the rebuilding; the second, the L1 penalty, charges $\lambda$ for every unit of latent activity.

## Why the charge finds the features

Every $d_j$ has length 1, so by the triangle inequality

$$ \Big\lVert \sum_j f_j d_j \Big\rVert \le \sum_j f_j $$

with equality only when all the arrows that fire point the same way. So of all exact rebuildings of an activation $x u_i$ (take $b = 0$), the cheapest uses only arrows along $u_i$: one latent with $d_j = u_i$, charged $\lambda x$. Two latents $30^\circ$ either side of $u_i$ would each have to report, to rebuild it exactly, $x / (2\cos 30^\circ) = 0.58 x$, and pay $1.15 \lambda x$ in all. So while features are rarely present together, the charge favours a list with an arrow on each feature. At $\lambda = 0$ nothing ties an arrow to a feature.

## Shrinkage

The charge also makes latents report too little. Take the activation $x u_i$, a latent with $d_j = u_i$, the others silent and $b = 0$. The latent's best report $f$ makes $(x - f)^2 + \lambda f$ smallest; setting its slope $-2(x - f) + \lambda$ to zero gives

$$ f = x - \lambda / 2 $$

or $0$ when $x < \lambda / 2$. Example: $\lambda = 0.3$ and $x = 0.8$ give $f = 0.65$, about 81 % of the truth. At $\lambda = 2$ every $x \le 1$ gives $f = 0$: nothing fires.

## The scores

- Variance left unexplained: the average of $\lVert h - \hat h \rVert^2$, over the average squared distance of the activations from their mean.
- L0: how many latents fire for one activation, on average.
- A feature is found when a latent that is not dead has an arrow within $18^\circ$ of it (cosine above 0.95) and, shown the features one at a time, gives at least three quarters of its response to that one.

# In the picture

- Dots: the activations $h$, all the autoencoder is given. Dashed numbered spokes: the true features $u_i$.
- Each latent has a colour and a letter; its bold arrow is $d_j$. A thin line ending in a dot, when shown, is $e_j$.
- The matrix: one row per latent, one column per feature, bright where $d_j$ points along $u_i$. The bars beside it: how often each latent fires.
- The scatter: present features, true size $x_i$ across, what their latents report up. The dashed line is an exact report; shrinkage sags below it.
- The charts: variance left unexplained, and L0 against the dashed line at $n(1 - S)$.
- Red: what the rebuilding misses of the ringed activation, $h - \hat h$.

> Try first "five features in a plane", then "no penalty ($\lambda = 0$)".

see: superposition, sensing, pca, transformer
