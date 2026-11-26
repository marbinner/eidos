# The idea

A network often has more things to keep track of than it has neurons. This scene trains a tiny network to show what happens then: $n$ features are squeezed through $m$ hidden dimensions, fewer than $n$, and rebuilt on the other side. Each feature gets a direction, an arrow, and with more arrows than dimensions they cannot all stand at right angles. You need dot products, the transpose of a matrix, and training to lower a loss.

# The mathematics

## The model

An input is a list of $n$ numbers $x_1, \ldots, x_n$, the features. Each is $0$ with probability $S$, the sparsity, and otherwise a random number between 0 and 1. An $m \times n$ matrix $W$ squeezes the input, and its transpose reads it back:

$$ h = W x, \qquad x' = \operatorname{ReLU}(W^\top h + b) $$

$b$ holds $n$ biases, and $\operatorname{ReLU}$ turns every number below zero into zero. Column $i$ of $W$, written $W_i$, is the arrow of feature $i$: since $h = x_1 W_1 + \cdots + x_n W_n$, the input is stored as a sum of arrows. Training lowers the weighted squared error, averaged over inputs:

$$ L = \sum_i I_i \, (x_i - x'_i)^2 $$

$I_1 = 1$, and each next importance $I_i$ is the slider's factor times the one before.

## The overlaps

Together, $x' = \operatorname{ReLU}(W^\top W x + b)$, and entry $(i, j)$ of $W^\top W$ is the dot product $W_i \cdot W_j$. Switch feature $k$ on alone, at 1, and output $i$ receives $W_i \cdot W_k + b_i$ before the ReLU. Its own output gets $|W_k|^2 + b_k$, about 1 for a kept feature. Every other output gets a leak unless its arrow is at right angles to $W_k$. A leak below zero is cut off by the ReLU at no cost; a leak above zero has to be pushed down by a negative bias.

Example: in a regular pentagon of unit arrows, feature 1 alone sends $\cos 72^\circ = 0.31$ to its two neighbours, for a bias to push down, and $\cos 144^\circ = -0.81$ to the other two, for the ReLU to remove.

## Why sparsity decides

Packing works while features come singly: take opposite arrows $W_1 = (1, 0)$ and $W_2 = (-1, 0)$, with $b = 0$. The input $x = (0.7, 0)$ gives $h = (0.7, 0)$ and $x' = (0.7, 0)$: exact. The input $(0.7, 0.5)$ gives $h = (0.2, 0)$ and $x' = (0.2, 0)$: both wrong, because the two features cancel. The more rarely two features are present together, the more arrows can share the dimensions.

## A share of a dimension

The share of a dimension that feature $i$ has to itself is its squared length over everything that lies along its direction:

$$ D_i = \frac{|W_i|^2}{\sum_j (\hat W_i \cdot W_j)^2} $$

where $\hat W_i = W_i / |W_i|$ is its arrow cut to length 1, and the sum includes $j = i$. An arrow at right angles to all the others has $D_i = 1$; one of an opposite pair of equal length, at right angles to the rest, has $D_i = 1/2$. A corner of the pentagon has $1 + 2\cos^2 72^\circ + 2\cos^2 144^\circ = 2.5$ below the line, so $D_i = 2/5$: five corners fill $5 \cdot 2/5 = 2$ dimensions.

# In the picture

- Each numbered arrow is a column $W_i$; the brightest is the most important feature, and a faint stub one the network has dropped.
- The axes are the hidden dimensions, which are the neurons: an axis with several arrows leaning along it is a polysemantic neuron.
- The square of cells is $W^\top W$: red where two arrows point the same way, blue where opposite ways, pale at right angles. Beside it, the bias $b$.
- The coloured bars are the shares $D_i$; the curve at the bottom is the loss $L$.
- The lowest bars switch one feature $k$ on alone: pale is $W_i \cdot W_k$, what arrives; solid is $x'_i$, what comes out.

> Try first "pentagon": five arrows settle $72^\circ$ apart, each with a share of $2/5$.

see: vectors, mlp, sae, packing
