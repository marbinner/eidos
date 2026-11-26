# The idea

A network with one input $x$, one hidden layer and one output can follow any continuous curve on a closed interval, given enough neurons and an activation that is not a polynomial. The reason is plain: each hidden neuron adds one simple piece, a hinge or a smooth step, and the output adds the pieces up. This scene fits a curve that way and draws every piece. You need the playground's neurons and gradient descent.

# The mathematics

## A sum of pieces

With $n$ hidden neurons, neuron $j$ having weights $w_j$ and $v_j$ and bias $b_j$, and the activation $\sigma$, the network's output is

$$ \hat y(x) = c + \sum_{j=1}^{n} v_j \, \sigma(w_j x + b_j) $$

Each term $v_j \sigma(w_j x + b_j)$ is one thin line: the shape of $\sigma$, moved and stretched sideways by $b_j$ and $w_j$, and scaled by $v_j$. With ReLU, $\sigma(z) = \max(0, z)$, it is 0 on one side of $x = -b_j / w_j$ and a straight line on the other: a hinge. There, and nowhere else, this piece changes the slope of $\hat y$, by $v_j |w_j|$. So $n$ ReLU neurons make a broken line with at most $n$ bends. With $\tanh$ the piece is a smooth step of height $2|v_j|$ centred at the same place, steeper for larger $|w_j|$.

## Why enough pieces follow any curve

Mark points close together along a continuous curve on $[-3, 3]$ and join them with straight segments. Because the curve is continuous, the broken line follows it as closely as you like once the points are close enough. And every broken line is a sum of hinges: a straight line plus, at each corner, one hinge whose $v_j |w_j|$ is the change of slope there. (A straight line on $[-3, 3]$ is itself a hinge at $x = -3$ plus a constant.)

Example: three hinges make a tent,

$$ \max(0, x + 1) - 2 \max(0, x) + \max(0, x - 1) $$

It is 0 up to $x = -1$; at $x = 0$ it is $1 - 0 + 0 = 1$; at $x = 0.5$, $1.5 - 1 + 0 = 0.5$; at $x = 2$, $3 - 4 + 1 = 0$, and it stays 0 beyond 1. Its slope changes by $+1$, $-2$ and $+1$ at $-1$, 0 and 1. Tents side by side join the dots of any curve.

The same holds for $\tanh$ and for every continuous activation that is not a polynomial: on a closed interval, one hidden layer with enough neurons comes as close to any continuous curve as you like. This theorem, universal approximation, says that such weights exist. Training still has to find them, and it can get stuck.

## The exception: polynomials

With $\sigma(z) = (z^2 - 1)/\sqrt{2}$ every piece is a parabola in $x$, and parabolas add up to a parabola:

$$ c + \sum_{j} v_j \frac{(w_j x + b_j)^2 - 1}{\sqrt{2}} = A x^2 + B x + C $$

for some numbers $A$, $B$, $C$, however many neurons there are. So the network can do no better than the best parabola. The wave $\sin(2x) + 0.3x$ is odd, $f(-x) = -f(x)$, and on points spread evenly either side of 0 the best parabola for an odd curve has $A = C = 0$: a straight line. The cubic $(z^3 - 3z)/\sqrt{6}$ likewise gives only cubics.

## The error

The network is trained, by Adam, to make the mean squared error over $N = 160$ points $x_i$ spread evenly over $[-3, 3]$ small:

$$ \frac{1}{N} \sum_{i=1}^{N} \bigl(\hat y(x_i) - y_i\bigr)^2 $$

# In the picture

- The thick grey curve is the target, and the cyan curve is $\hat y$.
- Each thin line is one neuron's piece $v_j \sigma(w_j x + b_j)$; the cyan curve is their sum plus $c$.
- With ReLU, the dots on the axis are the hinges, at $x = -b_j / w_j$.
- The small chart is the error, on a log scale.

> Try first "three hinges", then "sixty-four hinges": the same wave, built from 3 hinges and from 64.

see: mlp, hermite_expand, particles
