# The idea

Training a network is a walk downhill on its loss, a surface over the space of all its weights. This network has 97 weights, so the surface lives in 97 dimensions and cannot be drawn whole. A flat two-dimensional slice through it can, and the scene takes the slice that best contains the path the optimiser actually took. You need the playground's loss, gradient descent, and planes spanned by two directions.

# The mathematics

## The loss

Two hidden layers of 8 $\tanh$ neurons take one input $x$ and give $\hat y(x)$. Write $\theta$ for the list of all their weights and biases: $(8 + 8) + (8 \cdot 8 + 8) + (8 + 1) = 97$ numbers. The loss is the mean squared error on 48 points of the curve $y = \sin(1.7x) + 0.3x$, evenly spread over $[-2.5, 2.5]$:

$$ L(\theta) = \frac{1}{48} \sum_{i=1}^{48} \bigl(\hat y(x_i) - y_i\bigr)^2 $$

An optimiser starts at $\theta_0$ and takes steps $\theta_1, \theta_2, \ldots$ to the end $\theta_K$ (400 steps at first).

## A slice through 97 dimensions

Pick two directions $d_1$ and $d_2$ in the space of weights, each of length 1 and at right angles to each other. The plane through the end $\theta_K$ that they span has points $\theta_K + t_1 d_1 + t_2 d_2$, and the surface drawn is the true loss there:

$$ \ell(t_1, t_2) = L(\theta_K + t_1 d_1 + t_2 d_2) $$

Every point of the path is drawn at its shadow, the nearest point of the plane, at the height $\ell$ has there. That is the loss at the shadow, not the loss the optimiser had, unless the point lies in the plane; the end always does.

## The plane that fits the path best

The directions come from the path itself. Take the offsets $\theta_k - \theta_K$ of the points of the path from its end, and choose $d_1$ and $d_2$ to keep as much of their squared length as possible in the plane: the two principal directions of the offsets. The share kept is "path captured by this plane":

$$ \frac{\sum_k |\text{shadow of } (\theta_k - \theta_K)|^2}{\sum_k |\theta_k - \theta_K|^2} $$

Example, in three dimensions instead of 97: offsets $(3, 0, 0)$, $(0, 2, 0)$ and $(0, 0, 1)$ from the end. Their squared lengths along the axes are 9, 4 and 1, so the best plane is the one of the first two axes, and it captures $(9 + 4)/(9 + 4 + 1) = 13/14$, about 93 %. The third point's shadow falls on the end itself, and there the drawing shows the end's loss, not its own.

## Walls and floor

A fast drop and then a crawl is what gradient descent does in a long, narrow valley. Take the valley $5u^2 + 0.05v^2$, steep across ($u$), gentle along the floor ($v$). One step of gradient descent with step $\eta$ multiplies $u$ by $1 - 10\eta$ and $v$ by $1 - 0.1\eta$. With $\eta = 0.1$ the drop across the valley is done in one step, but $v$ shrinks by only 1 % per step. With $\eta$ above 0.2 the factor for $u$ is below $-1$: each step leaps to the far wall, higher than before. The walls limit the step, so the floor is crossed slowly. Momentum lets steps along the floor pile up, since there the gradient keeps its sign; Adam divides each weight's step by the size of its own recent gradients, so a weight on a steady gentle slope still moves about $\eta$ per step.

# In the picture

- Height and colour are $\ell$, yellow high and purple low, on a compressed scale (roughly the loss to the power 0.3) so that walls and floor both show.
- The white ball is the start's shadow; the cyan line and ball are the path's shadow and where the replay is; the ring marks the end.
- The read-outs give the true losses at the start and the end, and the share of the path captured.

> Try first "Adam", then "plain gradient descent": the same number of steps, and compare the loss at the end.

see: mlp, descent, pca, race
