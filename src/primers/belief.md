# The idea

A hidden Markov model is a process whose state nobody sees, here the positions of dials, and which emits tokens that depend on it. A Bayes filter forecasts each dial as well as possible from them. This scene points a Jacobian lens at the filter: had a glimpse shown something slightly different, how would the forecast respond later? The answer has the three factors of "What averaging Jacobians keeps" (read it first).

# The mathematics

## The dials and the filter

Each dial sits at one of $n$ sites round a ring. At every token it carries on with probability $\lambda$, the persistence: it moves $v$ sites on, a turn of $\theta = 360^\circ \cdot v / n$, and slips one site forward or back with probability $q$ each. Otherwise it jumps to any site, all equally likely. A token shows every dial's site with probability $\varepsilon$ (a glimpse), and no site otherwise.

The forecast gives each site a probability for the dial's next position. The scene draws it as an arrow $m$: the average, under the forecast, of the unit arrow pointing at the dial's site. A glimpse is exact and an empty token says nothing, so the filter starts from the site last shown and pushes it on token by token, each step multiplying $m$ by the same $\alpha$ (as complex numbers, multiplying by $e^{i\theta}$ turns an arrow by $\theta$):

$$ \alpha = \lambda \, c \, e^{i\theta}, \qquad c = 1 - 2q + 2q \cos(360^\circ / n) $$

Why: after a jump the arrows to all the sites average to zero; otherwise every arrow turns by $\theta$, and slipping averages it with its neighbours $360^\circ / n$ either side, which multiplies it by the factor $c$.

## Chains and factors

Nudge what a glimpse showed. The forecast $\tau$ tokens later moves by $\alpha^{\tau + 1}$ times the nudge until the next glimpse, which replaces everything, and by nothing after it. Laid tip to tail over lags $\tau = 1, \ldots, H$, they make a chain. A dial may have a key, $\pm 1$ at each token with average $\psi$; a glimpse right after a $-1$ shows it half a turn off and reverses the chain. A context is what varies: when the next glimpse comes, and the key.

- magnitude $a$: the chain's path length, on average;
- lag coherence $\rho_\tau$: the average start-to-end distance, over $a$;
- context coherence $\rho_c$: the length of the average end, over the average start-to-end distance;
- lens weight $\sigma = a \cdot \rho_\tau \cdot \rho_c$: the length of the average end, its singular value in the lens.

Lag $\tau$ is reached without a glimpse with probability $(1 - \varepsilon)^\tau$. With $r = |\alpha| (1 - \varepsilon)$ and an endless window, a geometric series gives

$$ a = |\alpha| \sum_{\tau \ge 1} r^\tau = \frac{|\alpha| \, r}{1 - r} $$

The average end is the same series, with the turning and the key kept in:

$$ \sigma = \frac{|\psi| \cdot |\alpha| \cdot r}{\left|1 - r \cdot e^{i(\varphi - \omega)}\right|} $$

Here $\varphi$ is the forecast's turn per token, either way round ($0^\circ$ to $180^\circ$), $\omega$ the lens's tuning ($0$ for the plain lens), and $\psi = 1$ with no key.

## Worked example

The yellow dial $\mathrm{B}$ starts with $n = 8$, $v = 2$, $\lambda = 0.9$, $q = 0.25$, $\varepsilon = 0.2$: so $\theta = 90^\circ$, $c = 0.5 + 0.5 \cos 45^\circ = 0.8536$, $|\alpha| = 0.7682$, $r = 0.6146$ and $a = 0.4721 / 0.3854 = 1.22$. As $|1 - 0.6146\, i| = 1.174$, $\sigma = 0.4721 / 1.174 = 0.40$. The pink dial $\mathrm{C}$ stands still, so its chains all point one way: $\sigma = a = 1.22$.

# In the picture

- Top: the rings. White circle: where a dial is; discs: the forecast; arrow: $m$.
- The strip: the tokens; lit ones are glimpses.
- "one nudge": a chain. "many nudges": where chains end, and their average end (white), of length $\sigma$.
- The bars: $a$, $\rho_\tau$, $\rho_c$ and $\sigma$ per dial, from the formulas; white marks are measured.
- Bottom: $\sigma$ against the tuning $\omega$, and against the studied dial's $\lambda$.

> Try first "memory: the magnitude sweep", where only the chains' lengths differ.

see: bayes, coherence, lyapunov
