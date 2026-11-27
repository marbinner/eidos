# The idea

Many networks have normalisation layers, which rescale what passes through them to a standard size. The weights that feed such a layer can be made longer or shorter without changing what the network computes: only their direction matters. This scene follows what gradient descent and weight decay do to such weights: two in a made-up loss, then a small network's. You need gradients, and a vector's length and angle.

# The mathematics

## Weights whose length does not matter

Suppose the loss $L$ does not change when the weights $w$ are multiplied by any $c > 0$, so $L(cw) = L(w)$. Two facts follow. The gradient $g$ is at right angles to $w$, since moving along $w$ only changes its length:

$$ w \cdot g = 0 $$

And it shrinks as $w$ grows: the gradient at $cw$ is $g/c$. So $G = |g|\,|w|$, the unit gradient, does not depend on the length. In the scene's network each hidden layer feeds a batch normalisation, and this holds up to a small guard $\varepsilon$ in it.

## One step, by Pythagoras

A step of gradient descent with step size $\eta$ and weight decay $\lambda$ is $w' = (1 - \eta\lambda)\,w - \eta g$. Its two parts are at right angles, so

$$ |w'|^2 = (1 - \eta\lambda)^2\,|w|^2 + \eta^2 |g|^2 $$

Without decay the length can only grow. The step also turns $w$ by an angle $\Delta$, and with $|g| = G/|w|$,

$$ \tan\Delta = \frac{\eta}{|w|^2} \cdot \frac{G}{1 - \eta\lambda} $$

The direction moves as if the step size were $\eta/|w|^2$: the effective learning rate.

Example: $|w| = 2$, $G = 1$ (so $|g| = 0.5$), $\eta = 0.1$, $\lambda = 0.01$. Then $|w'|^2 = 0.999^2 \times 4 + 0.01 \times 0.25 = 3.9945$, so $|w'| = 1.9986$, and $\tan\Delta = 0.05/1.998$, a turn of $1.43°$.

## Without decay, and with it

Without decay, $|w|^4$ grows by about $2\eta^2 G^2$ a step. If $G$ stays about the same, after many steps $t$ the turn per step is about $1/\sqrt{2t}$ radians, whatever $\eta$: learning slows as if its rate were decayed.

With decay the length settles where growth and shrinkage balance. Setting $|w'| = |w|$ above gives

$$ |w|^4 = \frac{\eta\,G^2}{\lambda\,(2 - \eta\lambda)} $$

Wherever the length is steady, $w' \cdot w = (1 - \eta\lambda)\,|w|^2$ gives exactly

$$ \cos\Delta = 1 - \eta\lambda, \qquad \Delta \approx \sqrt{2\eta\lambda} $$

whatever the gradient: $2.56°$ a step for $\eta = 0.1$ and $\lambda = 0.01$. Only the product $\eta\lambda$ sets this turn: weights $c$ times as long, stepped with $c^2\eta$ and $\lambda/c^2$, turn exactly as before. On such weights decay does not change what the network can compute; it keeps them short, and the effective learning rate up.

## Adam, Muon, and a ball

Adam's step does not change when every gradient is multiplied by one positive number ($\varepsilon$ aside). Decay added to the gradient, $g + \lambda w$ (L2), goes through Adam's division by $\sqrt{\hat v}$ with the rest; decay taken off the weights, $w \leftarrow (1 - \eta\lambda)\,w$ (AdamW), does not. For gradient descent the two are the same rule.

If each step is $\eta u$ with $u$ of size at most $b$ in some norm, and the decay is taken off the weights with $\eta\lambda \le 1$, the triangle inequality gives

$$ |w'| \le (1 - \eta\lambda)\,|w| + \eta b $$

so once the start is forgotten, $|w| \le b/\lambda$. For a sign-like step, every entry at most 1 in size, each weight stays within $1/\lambda$. Muon's step on a square layer, with its reference polynomial, is $\eta$ times a matrix whose singular values are at most 1.2024, so the layer's largest singular value stays within $1.2024/\lambda$.

# In the picture

- The plane: the rays are lines of equal loss, the brightest the lowest. Cyan is the path with decay, grey the same pushes without; the dashed circle is the balance.
- The charts: length and turn per step against the step, on a logarithmic scale; dashed, the formulas: the balance, $\arccos(1 - \eta\lambda)$ and $1/\sqrt{2t}$.
- The network: one line per hidden layer, lighter further in; orange is Adam, blue Muon, yellow Muon's ball.

> Try first "two weights: a spiral and a circle", then "half the step, twice the decay".

see: normgrad, adam, muon, eos
