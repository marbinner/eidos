# The idea

Nudge the input of a deep network a little and follow the nudge through the layers. Each layer stretches it in some directions and squashes it in others, and over many layers the factors multiply up. The Lyapunov exponents are the rates of growth per layer: depth plays the part that time plays in a system that evolves. You need matrices, the chain rule, logarithms and singular values (the SVD scene).

# The mathematics

## The network

The state $h$ is a list of $n$ numbers ($n = 24$ to begin with). Each of $L$ layers ($L = 30$) makes a new state from the last:

$$ h' = \varphi(W h) $$

with $\varphi = \tanh$ applied to each number, and $W$ a fresh $n \times n$ matrix of random numbers for each layer, of typical size $g / \sqrt{n}$. In the residual form a layer adds to the state instead: $h' = h + \beta \varphi(W h)$.

## Jacobians

A small nudge $\delta$ to the state before a layer moves the state after it by $J_l \delta$, where $J_l$ is that layer's Jacobian. By the chain rule,

$$ J_l = D_l W, \qquad D_l = \diag\big(\varphi'(W h)\big), \qquad \varphi'(z) = 1 - \tanh^2 z $$

so row $i$ of $W$ is scaled by the slope of neuron $i$: a neuron pushed far into the flat part of $\tanh$ passes almost nothing on. (In the residual form, $J_l = I + \beta D_l W$.) The nudge passes through every layer in turn, so for the whole network

$$ J = J_L \cdots J_2 J_1 $$

and, with $\tanh$, $J$ depends on the input, because the slopes do.

## Exponents

Let $s_1 \ge s_2 \ge \cdots \ge s_n$ be the singular values of $J$, and $v_1$ the input direction it stretches most. A small ball of nudges comes out as an ellipsoid with half-axes $s_i$ times the ball's radius. The exponents turn these into rates per layer:

$$ \chi_i = \frac{\ln s_i}{L} $$

Above zero a nudge along that direction grows; below zero it dies. Example: if the best nudge comes out $673.7$ times as large after 30 layers, then $\chi_1 = \ln 673.7 / 30 = 6.513 / 30 = 0.217$. If instead each of 3 layers doubled one axis of a plane and halved the other, $s_1 = 8$ and $s_2 = 1/8$, so $\chi_1 = \ln 2 = 0.69$ and $\chi_2 = -0.69$.

## Order, chaos and the edge

The scene prints a rule of thumb for $\chi_1$:

$$ \chi_1 \approx \tfrac{1}{2} \ln\big(g^2 \cdot \operatorname{mean} \varphi'^2\big) $$

The rough reasoning behind it, not a proof: in a wide layer, multiplying by $W$ multiplies the squared size of a typical nudge by about $g^2$, and the slopes then multiply it by about the average of $\varphi'^2$. The size itself grows by the square root of that, which halves the logarithm. A small $g$ gives $\chi_1 < 0$, order: every nudge dies. A large $g$ gives $\chi_1 > 0$, chaos. Between them, at $\chi_1$ near 0, is the edge.

## Averaging Jacobians

At bottom right the scene finds $J$ at 12 other inputs, scales each so that its $s_1$ is 1, and averages them. If they were all the same, the average would keep $s_1 = 1$. If they differ, they partly cancel: $\diag(1, 0)$ and $\diag(0, 1)$ average to $\diag(1/2, 1/2)$, which keeps 50 %. Their overlap, the average cosine between two Jacobians read as lists of $n^2$ numbers, is 0 for that pair and 1 for Jacobians that are the same.

# In the picture

- Top left, $\ln$ of a nudge's size after each layer: yellow along $v_1$ (its slope from start to end is $\chi_1$), cyan in six random directions, pink in the fastest-dying direction. The green dots are a real nudge of size 0.1.
- The bars: all $n$ exponents $\chi_i$, orange above zero, blue below.
- Top right: a small circle of nudges in the plane of $v_1$ and $v_2$, after the layer chosen: an ellipse.
- Bottom right: $\ln(s_i / s_1)$ of the Jacobian at each of 12 inputs (thin cyan), and $\ln s_i$ of their average (thick pink).

> Try first "Chaos: gain 3", then "Order: gain 0.7".

see: svd, powers, signal, coherence
